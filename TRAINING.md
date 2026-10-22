# Training on the cleaned oopsie subset

Phase B materializes a bounded, immediately-trainable slice of the cleaned oopsie
corpus and stages it to S3 in **two** formats. Point either trainer at the S3 paths
below.

```
s3://tri-ml-datasets-uw2/anthonyliang/oopsie_clean/
    manifest.parquet            # one row per clean episode (+ manifest.jsonl)
    lerobot/                    # LeRobotDataset v2.1  (repo_id local/oopsie_clean_subset)
    wds/
        Charles_UFAL/shards/    # one dataset PER LAB:
        BONN_MB/shards/         #   shard_*.tar + manifest.jsonl + stats.json
        Araya/shards/           #   + preprocessing_config.yaml (train-time reads it)
        Enatic_OpenArm/shards/
        CMU_Aire/shards/
```

WDS is exported **one dataset per lab** because it is the disk-safe streaming unit
(see "Reproducing" below) AND the natural train-time unit: vla_foundry's
`dataset_manifest` / `dataset_statistics` / `dataset_weighting` / `dataset_modality`
are parallel lists, each lab self-resolves its shards from its own manifest dir, and
the per-lab `stats.json` are pooled by the normalizer at load time.

The subset spans five embodiments / action spaces (common-18 padded + per-dim
`action_mask` / `action_dim_valid`):

| lab            | embodiment               | action space        | arms | episodes |
|----------------|--------------------------|---------------------|------|----------|
| Charles_UFAL   | so101_single             | joint 5 + grip 1    | 1    | 30       |
| BONN_MB        | ur5_single               | joint 6 + grip 1    | 1    | 30       |
| Araya          | franka_cartesian_single  | cartesian 7 + grip 1| 1    | 30       |
| Enatic_OpenArm | openarm_biarm            | joint 16 + grip 2   | 2    | 30       |
| CMU_Aire       | xarm7_biarm              | cartesian 14 + grip 2| 2   | 15       |

(135 clean episodes total. Caps are deliberately low and materialization is streamed
one lab at a time; raise `configs/subset.yaml:labs` to scale.)

Metadata-conditioning prefix (prepended to the instruction, CFG dropout 0.3):

```
<opt=success|failure> <speed=0|1|2> <emb=NAME> <op=OPERATOR> 
```

---

## pi05 / openpi  (config `pi05_oopsie_clean_subset`)

Runs from the openpi checkout (`dsrl_openpi_flip`). The config is the metadata-
conditioned `pi05_oopsie_smoke` stack (MetadataPrefixPrompt + OopsieInputs/Outputs,
COMMON_DIM=18 + `action_mask`), repointed at `local/oopsie_clean_subset` with a real
train budget (30k steps, batch 32).

```bash
cd ~/dsrl_openpi_flip

# 1. Pull the LeRobot dataset from S3 into HF_LEROBOT_HOME.
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-~/.cache/huggingface/lerobot}
aws s3 sync s3://tri-ml-datasets-uw2/anthonyliang/oopsie_clean/lerobot/ \
    "$HF_LEROBOT_HOME/local/oopsie_clean_subset/"

# 2. Compute norm stats (writes under assets_dirs/local/oopsie_clean_subset).
uv run python scripts/compute_norm_stats.py --config-name pi05_oopsie_clean_subset

# 3. Train.
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
    uv run python scripts/train.py pi05_oopsie_clean_subset --exp-name oopsie_clean_v1
```

DGX gotchas:
- **`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`** — without it JAX pre-grabs too little /
  fragments and OOMs mid-run on the 80 GB cards.
- **pyav / torchcodec video backend** — LeRobot auto-selects `torchcodec` when it is
  importable, but torchcodec's native lib needs CUDA runtime libs (e.g.
  `libnvrtc.so.*`) that are missing on CPU-only or mismatched-CUDA boxes, so decode
  raises `Could not load libtorchcodec`. Force pyav. Newer lerobot builds ignore
  `LEROBOT_VIDEO_BACKEND`, so set the backend on the dataset object directly
  (`dataset.video_backend = "pyav"`), as `exporters/verify_lerobot.py` does; older
  builds still honor `export LEROBOT_VIDEO_BACKEND=pyav`. The dataset is encoded with
  SVT-AV1; pyav decodes it fine.

Proven smoke: the identical transform stack on the 3-lab `pi05_oopsie_smoke` dataset
trained clean on DGX — **loss 0.356 → 0.171** over the 200-step smoke — so the
metadata prefix, the common-18 padding, and the `action_mask` loss masking are all
wired correctly. The clean-subset config only swaps the dataset and raises the budget.

---

## vla_foundry Qwen  (config `qwen_oopsie_clean_subset`)

Runs from the `vla_foundry` checkout. The config mirrors `qwen_oopsie_metacond`
(metadata-conditioning enabled, `mask_padded_action_dims: true`, canonical 3-cam
layout, action/proprio dim 18), repointed at the S3 WDS shards.

Local / single-node:

```bash
cd ~/vla_foundry
./.venv/bin/python -m vla_foundry.main \
    --config_path vla_foundry/config_presets/training_jobs/qwen_oopsie_clean_subset.yaml
```

SageMaker:

```bash
cd ~/vla_foundry
./.venv/bin/python sagemaker/launch_training.py \
    --config_path vla_foundry/config_presets/training_jobs/qwen_oopsie_clean_subset.yaml
```

For the real job, raise `hparams.global_batch_size` / `per_gpu_batch_size`,
`total_train_samples`, and set `distributed.world_size` / `hparams.world_size` to the
instance GPU count. `resume_from_checkpoint` is null on purpose — the 14-D YAM cotrain
base cannot warm-start an 18-D action head.

The WDS shards carry `stats.json` with `percentile_1_per_timestep` /
`percentile_99_per_timestep` of shape (T=16, C=18), which the trainer's
`percentile_1_99 / per_timestep` normalization consumes directly.

---

## Reproducing the materialization (from this repo)

The materialization is **disk-safe and streamed**: it stages ONE lab, appends it to
the consolidated LeRobot dataset, exports that lab's own WDS dataset, uploads both to
S3, deletes the local copy, and only then moves to the next lab. The prior run filled
puget's root disk by accumulating ~98 GB of per-frame-JPEG WDS tars for all labs
locally before any upload; the streamer keeps local scratch under
`configs/subset.yaml:phaseB2_cap_gb` (default 15 GB) and aborts if free disk drops
below `df_floor_gb` (80 GB).

```bash
# 1. manifest (one row per clean episode) — dsrl_openpi_flip uv env
cd ~/dsrl_openpi_flip
uv run --with boto3 python ~/oopsie_processing/run.py stage10

# 2. stream both formats to S3, one lab at a time (LeRobot import + boto3 here; the
#    WDS step shells out to vla_foundry's OWN venv internally for its ray workers)
uv run --with boto3 python ~/oopsie_processing/exporters/stream_to_s3.py
```

Verify the S3 artifacts load on CPU (no GPU, no training):

```bash
# LeRobot: pulls lerobot/ back from S3, runs the pi05 transform stack
cd ~/dsrl_openpi_flip
uv run --with boto3 python ~/oopsie_processing/exporters/verify_lerobot.py

# WDS: pulls one lab's shards back from S3, loads via vla_foundry's dataloader
#   (vla_foundry's OWN venv — do NOT wrap in `uv run`)
~/vla_foundry/.venv/bin/python ~/oopsie_processing/exporters/verify_wds.py
```

`configs/subset.yaml` controls the per-lab caps, path filters, outcome balancing, disk
guards, and all output paths. `configs/embodiments.yaml` is the shared action/camera
registry (stages 5/6 + both exporters read it — no silent defaults). The WDS export
depends on the five subset labs registered in
`vla_foundry/.../converters/oopsie.py:LAB_SPECS` (canonical camera maps) plus the
`qwen_oopsie_clean_subset.yaml` training config (5 per-lab S3 datasets); the pi05 side
depends on the `pi05_oopsie_clean_subset` TrainConfig in `dsrl_openpi_flip`. Those
edits live in their own repos (not committed here).

`exporters/materialize.py`, `export_lerobot.py`, and `export_webdataset.py` keep
standalone whole-subset `run()` entrypoints for debugging, but the whole-subset WDS
path is NOT disk-safe at scale — always use `stream_to_s3.py` for real materialization.
