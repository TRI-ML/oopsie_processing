"""CPU verify of the S3 WebDataset export through vla_foundry's own dataloader.

Pulls ONE lab's shards back FROM S3 (proving the per-lab manifest+stats+shards
round-trip and self-resolve), loads qwen_oopsie_clean_subset.yaml but repoints it at
that single local lab dataset, forces dropout_p=0 so the metadata prefix is always
present, builds the dataloader exactly like main.py, pulls one batch, and confirms:
  * action_dim_valid is a width-18 1-prefix (per sample);
  * the RAW stored action on padded dims is exactly 0 (read straight from a shard
    tar's lowdim.npz) -- the NORMALIZED batch value is a nonzero constant by design
    (pooled percentiles), which is precisely why it must be masked;
  * the resolved diffusion loss mask (future_mask x action_dim_valid) is 0 exactly on
    the padded dims;
  * the metadata prefix ("<opt=.. <speed=.. <emb=.. <op=.. ") survives tokenization.
No GPU. Needs network for S3 + the Qwen processor/tokenizer config (once).

Run under vla_foundry's OWN venv (do NOT wrap in `uv run`):

    /home/anthonyliang/Documents/vla_foundry/.venv/bin/python \
        /home/anthonyliang/Documents/oopsie_processing/exporters/verify_wds.py
"""
import glob
import io
import json
import os
import subprocess
import sys
import tarfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from oopsie_processing.common import config  # noqa: E402

from vla_foundry.params.train_experiment_params import load_experiment_params_from_yaml  # noqa: E402
from vla_foundry.data.dataloader import get_datastring_input, get_wds_dataloader  # noqa: E402
from vla_foundry.data.processor import get_processor  # noqa: E402
from vla_foundry.losses import resolve_loss_mask  # noqa: E402

CFG = ("/home/anthonyliang/Documents/vla_foundry/vla_foundry/"
       "config_presets/training_jobs/qwen_oopsie_clean_subset.yaml")

cfg_sub = config.load_merged("subset")
lab = config.require(cfg_sub, "verify_lab")
s3_wds = config.require(cfg_sub, "s3_wds_prefix").rstrip("/") + "/"
SHARDS = os.path.join(config.require(cfg_sub, "durable_dir"), "verify", "wds", lab, "shards")
os.makedirs(SHARDS, exist_ok=True)
src = f"{s3_wds}{lab}/shards/"
print(f"[verify-wds] pulling {src} -> {SHARDS}")
r = subprocess.run(["aws", "s3", "sync", src, SHARDS + "/", "--no-progress", "--only-show-errors"],
                   capture_output=True, text=True)
if r.returncode != 0:
    raise RuntimeError(f"s3 sync failed: {r.stderr.strip()[:300]}")
MANIFEST = os.path.join(SHARDS, "manifest.jsonl")
STATS = os.path.join(SHARDS, "stats.json")


def _set(obj, k, v):
    object.__setattr__(obj, k, v)


# --- A. RAW stored data: padded action dims are exactly 0 in the shard tars ---
tar = sorted(glob.glob(os.path.join(SHARDS, "shard_*.tar")))[0]
t = tarfile.open(tar)
keys = sorted({n.split(".")[0] for n in t.getnames()})
raw_checked = 0
for k in keys:
    meta = json.loads(t.extractfile(k + ".metadata.json").read())
    adv = meta["action_dim_valid"]
    valid = int(sum(adv))
    npz = np.load(io.BytesIO(t.extractfile(k + ".lowdim.npz").read()))
    akey = next(x for x in npz.files if "action" in x)
    arr = npz[akey]
    pad = np.abs(arr[:, valid:])
    maxpad = float(pad.max()) if pad.size else 0.0
    assert maxpad == 0.0, f"raw stored padded action dims must be 0 (got {maxpad})"
    if valid < 18:
        raw_checked += 1
    if raw_checked >= 4:
        break
print(f"[verify-wds] raw stored padded action dims == 0 "
      f"(checked {raw_checked} padded samples in {os.path.basename(tar)})")

# --- B. dataloader exactly like main.py, repointed at the one local lab dataset ---
cfg = load_experiment_params_from_yaml(CFG)
_set(cfg.data, "dataset_manifest", [MANIFEST])
_set(cfg.data, "dataset_statistics", [STATS])
_set(cfg.data, "dataset_weighting", [1.0])
_set(cfg.data, "dataset_modality", ["robotics"])
_set(cfg.data, "num_workers", 0)
_set(cfg.data, "shuffle_buffer_size", 8)
_set(cfg.data, "shuffle_initial", 4)
_set(cfg.data.metadata_conditioning, "dropout_p", 0.0)
_set(cfg.hparams, "global_batch_size", 4)
_set(cfg.hparams, "per_gpu_batch_size", 4)
_set(cfg.distributed, "use_distributed", False)
_set(cfg.distributed, "world_size", 1)
_set(cfg.distributed, "device", "cpu")
_set(cfg, "total_train_samples", 64)
_set(cfg, "num_checkpoints", 1)
cfg.resolve_derived_fields()

n_ds = len(cfg.data.dataset_manifest)
spc = cfg.total_train_samples // cfg.num_checkpoints
datastrings, nspd, _, _ = get_datastring_input(
    num_samples=spc,
    curr_shard_idx_per_dataset=[0] * n_ds,
    shard_shuffle_seed_per_dataset=[cfg.hparams.seed] * n_ds,
    manifest_paths=cfg.data.dataset_manifest,
    dataset_weighting=cfg.data.dataset_weighting,
    allow_multiple_epochs=cfg.data.allow_multiple_epochs,
    num_workers_per_gpu=cfg.data.num_workers,
    world_size=cfg.distributed.world_size,
)
di = get_wds_dataloader(datastrings, nspd, 0, cfg)
batch = next(iter(di.dataloader))
print("[verify-wds] batch keys:", sorted(batch.keys()))

adv = batch["action_dim_valid"]            # [B, D]
actions = batch["actions"]                 # [B, T, D] (NORMALIZED)
assert adv.shape[-1] == 18, f"action_dim_valid width {adv.shape[-1]} != 18"
for b in range(adv.shape[0]):
    m = adv[b].numpy()
    assert set(np.unique(m).tolist()) <= {0.0, 1.0}, f"mask not binary: {np.unique(m)}"
    valid = int(m.sum())
    assert np.all(m[:valid] == 1.0) and np.all(m[valid:] == 0.0), f"mask not a 1-prefix: {m.tolist()}"
    norm_pad = float(actions[b, :, valid:].abs().max()) if valid < 18 else 0.0
    print(f"[verify-wds] sample {b}: valid_dims={valid} mask={m.astype(int).tolist()} "
          f"(normalized padded max|val|={norm_pad:.3f} -> masked out)")

# --- C. loss mask: future_mask x action_dim_valid is 0 exactly on padded dims ---
fut = batch["future_mask"]
lm = resolve_loss_mask(fut, adv)
print(f"[verify-wds] loss mask shape={tuple(lm.shape)} future_mask shape={tuple(fut.shape)}")
for b in range(adv.shape[0]):
    valid = int(adv[b].sum())
    if valid < 18:
        assert float(lm[b, :, valid:].abs().max()) == 0.0, "loss mask nonzero on padded dims"
print("[verify-wds] loss mask is 0 on all padded action dims (per-sample)")

# --- D. metadata prefix survives tokenization ---
proc = get_processor(cfg.data)
tok = getattr(proc, "tokenizer", proc)
decoded = tok.batch_decode(batch["input_ids"], skip_special_tokens=False)
hit = 0
for b, txt in enumerate(decoded):
    has = all(mk in txt for mk in ("<opt=", "<speed=", "<emb=", "<op="))
    i = txt.find("<opt=")
    snippet = txt[i:i + 110] if i >= 0 else txt[:110]
    print(f"[verify-wds] decoded[{b}] prefix_present={has} :: {snippet!r}")
    hit += int(has)
assert hit == len(decoded), "metadata prefix missing from tokenized text in some samples"
print(f"[verify-wds] PASS: prefix tokenized {hit}/{len(decoded)}, 18-D 1-prefix mask, "
      f"raw padded dims zero + loss mask zeroed on padded dims")
