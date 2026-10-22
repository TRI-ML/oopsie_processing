"""CPU verify of the S3 LeRobot export through the openpi pi05 pipeline.

Pulls the exported LeRobot dataset back FROM S3 (proving the uploaded artifact is
self-contained), builds the pi05_oopsie_clean_subset data config, loads it via
openpi's create_torch_dataset, and runs the repack + data transforms
(MetadataPrefixPrompt + OopsieInputs) on CPU. Confirms: the metadata prefix lands in
the prompt, model-input shapes are right, and action_mask marks exactly the valid
dims (padding zeroed). No GPU, no norm stats, no tokenizer download required.

Run under the openpi uv env (has lerobot + boto3):

    cd /home/anthonyliang/Documents/dsrl_openpi_flip
    uv run --with boto3 python /home/anthonyliang/Documents/oopsie_processing/exporters/verify_lerobot.py
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from oopsie_processing.common import config  # noqa: E402

cfg = config.load_merged("subset")
repo_id = config.require(cfg, "lerobot_repo_id")
s3_lerobot = config.require(cfg, "s3_lerobot_prefix")
verify_home = os.path.join(config.require(cfg, "durable_dir"), "verify", "lerobot_home")
dst = os.path.join(verify_home, repo_id)
os.makedirs(dst, exist_ok=True)
print(f"[verify-lerobot] pulling {s3_lerobot} -> {dst}")
r = subprocess.run(["aws", "s3", "sync", s3_lerobot, dst + "/", "--no-progress", "--only-show-errors"],
                   capture_output=True, text=True)
if r.returncode != 0:
    raise RuntimeError(f"s3 sync failed: {r.stderr.strip()[:300]}")
os.environ["HF_LEROBOT_HOME"] = verify_home

import openpi.training.config as _config  # noqa: E402
import openpi.training.data_loader as _data_loader  # noqa: E402
import openpi.transforms as _transforms  # noqa: E402

tc = _config.get_config("pi05_oopsie_clean_subset")
data_config = tc.data.create(tc.assets_dirs, tc.model)
ds = _data_loader.create_torch_dataset(data_config, tc.model.action_horizon, tc.model)
print(f"[verify-lerobot] dataset episodes={ds._dataset.meta.total_episodes} "
      f"frames={ds._dataset.meta.total_frames}")
# This lerobot build auto-selects torchcodec when importable, but its native lib
# needs CUDA runtime libs absent on CPU-only boxes. Force the pyav decoder so the
# CPU verify actually decodes the SVT-AV1 videos (same as the DGX pyav fallback).
if hasattr(ds._dataset, "video_backend"):
    ds._dataset.video_backend = "pyav"

raw = ds[0]
print("[verify-lerobot] raw keys:", sorted(k for k in raw if not k.startswith("observation.images")))
am = np.asarray(raw["action_mask"]).reshape(-1)
act = np.asarray(raw["action"])
act = act.reshape(-1)[:18] if act.ndim == 1 else act[0, :18]
valid = int(am.sum())
print(f"[verify-lerobot] action_mask shape={am.shape} valid_dims={valid} mask={am.astype(int).tolist()}")
assert am.shape[0] == 18, "action_mask must be width-18"
assert set(np.unique(am).tolist()) <= {0.0, 1.0}
assert np.all(am[:valid] == 1.0) and np.all(am[valid:] == 0.0), "mask must be a 1-prefix"
pad_vals = np.abs(act[valid:])
print(f"[verify-lerobot] padded action dims [{valid}:18] max|val|={float(pad_vals.max()) if pad_vals.size else 0.0}")
assert pad_vals.size == 0 or pad_vals.max() == 0.0, "padded action dims must be zero"

repack = _transforms.compose(data_config.repack_transforms.inputs)
sample = repack(raw)
meta = _transforms.MetadataPrefixPrompt(dropout_p=0.0)
sample = meta(sample)
oi = next(t for t in data_config.data_transforms.inputs if t.__class__.__name__ == "OopsieInputs")
model_in = oi(sample)
prompt = sample["prompt"]
if isinstance(prompt, bytes):
    prompt = prompt.decode()
print(f"[verify-lerobot] prompt = {prompt!r}")
assert prompt.startswith("<opt="), "metadata prefix must lead the prompt"
assert "<emb=" in prompt and "<speed=" in prompt and "<op=" in prompt
st = np.asarray(model_in["state"])
print(f"[verify-lerobot] state shape={st.shape} image keys={sorted(model_in['image'])}")
assert st.shape[-1] == 18
assert set(model_in["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
for k, v in model_in["image"].items():
    print(f"   image[{k}] shape={np.asarray(v).shape} dtype={np.asarray(v).dtype}")
print("[verify-lerobot] PASS: prefix in prompt, shapes 18/3-cam, action_mask 1-prefix + padded dims zeroed")
