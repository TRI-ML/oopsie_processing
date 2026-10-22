# oopsie preprocessing — handoff

Handoff of the oopsie dataset cleaning/standardization work. Everything you need is
either in this repo or at the S3 / URLs below. Status as of 2026-10-22.

## TL;DR

- **Raw corpus** and the **cleaned-subset output** both live in S3 (below).
- **Phase A (audit/inventory over all 61 labs)** is done — scope, dedup, quarantine,
  schema validation, alignment. Reports are machine-readable.
- **Phase B (materialize → trainable formats)** is done **only as a 5-lab / 135-episode
  proof subset** (both LeRobot and WebDataset). Scaling to the full corpus is the main
  remaining task.
- **Phase C (wire the manifest into pi05 / vla_foundry training)** exists as smoke-tested
  converters but they are **uncommitted local files** (see "Not yet committed").

## Where everything is

### This repo — `TRI-ML/oopsie_processing` (private, branch `main`, HEAD `ebce4e5`)

| Path | What |
|---|---|
| `PLAN.md` | The full 10-stage non-destructive cleaning plan + phasing. Start here. |
| `README.md` | Scope + principle (non-destructive). |
| `src/oopsie_processing/` | Stage code: `stage1_taxonomy` … `stage10_manifest`, `stage7_alignment`, `data_quality`, plus `common/` (s3 streaming, h5 io, sampling, reports). |
| `exporters/export_lerobot.py` | Phase B → LeRobot v2.1 (for pi05). Disk-safe (streams to S3). |
| `exporters/export_webdataset.py` | Phase B → WebDataset tar shards (for vla_foundry). |
| `configs/*.yaml` | Per-stage config (no silent defaults). |
| `run.py` | Stage runner. |
| `QUALITY_CHECKS.md` | Phase A QC results write-up. |
| `MANIFEST_SCHEMA.md` | The per-episode manifest columns (one row per cleaned episode). |
| `TRAINING.md` | How to load the cleaned data in pi05 / vla_foundry + the gotchas. |

### S3 (us-west-2, bucket `tri-ml-datasets-uw2`)

- **Raw (read-only, source of truth):** `s3://tri-ml-datasets-uw2/anthonyliang/oopsie/`
  — 1.89 TB, 490,288 objects, 84 prefixes, **61 real labs**, ~82,182 episodes.
  Schema `oopsiedata_format_v1`: one small `.h5` per episode (lowdim only; actions +
  proprio) + one `.mp4` per camera (h5 holds path strings, no inline images).
- **Cleaned subset output (Phase B):** `s3://tri-ml-datasets-uw2/anthonyliang/oopsie_clean/`
  - `lerobot/` — 1.45 GB, 135 episodes / 154,598 frames, 3×224² AV1 video.
  - `wds/<lab>/shards/` — 16.9 GB WebDataset tars.
  - `manifest.parquet` + `manifest.jsonl` — one row per cleaned episode.
  - Subset = 5 labs: **Charles_UFAL** (so101), **BONN_MB** (ur5), **Araya** (franka
    cartesian), **Enatic_OpenArm** (openarm bimanual, full 18-D), **CMU_Aire** (dual
    xarm7 bimanual, 15 eps).

### Audit + viewer report

- <https://aliang8.github.io/reports/oopsie/> — password `oopsie`.
- Per-lab video gallery, a "what one cleaned episode looks like" section (3 worked
  examples incl. a raw manifest JSON row + camera→source-key tables), and the full
  non-destructive audit.

## What the pipeline produces (the contract)

Each cleaned episode becomes one uniform record regardless of its native action space or
camera naming:

- **Actions** padded to a common **18-D** vector + a per-frame `action_mask` marking the
  valid (unpadded) dims, keyed by embodiment. 18 = widest sampled embodiment (Enatic
  bimanual joint16 + gripper2).
- **Cameras** remapped to a canonical 3-slot layout (`base` / `left_wrist` /
  `right_wrist`), re-encoded 224×224. `camera_s3_keys` in the manifest keeps the actual
  source mp4 path per slot (traceable back to raw).
- **Metadata axes** for pi07-style conditioning, injected as discrete text-prefix tokens
  with CFG dropout: `<opt=success|failure>`, `<speed=0|1|2>`, `<emb=…>`, `<op=…>` +
  the language instruction. See `MANIFEST_SCHEMA.md` for all columns.
- `speed` = `mean(|Δaction| over valid dims) × control_freq`; `speed_bucket` = tertile
  over the conversion set (NB: **relative** to whatever episodes are converted — freeze
  the thresholds for a full-corpus run so buckets are stable).

## Status by phase

- **Phase A — done (all 61 labs, read-only).** Authoritative scope: 61 real labs (23
  non-lab prefixes excluded: origin/dup/preview/staging/15 empty placeholders).
  Quarantine: 15,554 paths (15,498 USP_MB `.hidden`, 50 UW_REAL `.h5.bak`, CMU_Auton
  261-byte mp4). Schema 99.65% pass (Stanford_REAL 43% — recoverable video-pointer prefix
  bug). **Alignment 91.3%** — 10 labs hold per-episode frame≠T misalignment (Araya,
  Harvard_EML, INRIA_SC, McGill_DM_HL, NUS_CLeAR, PSU_FMC, RIKEN_AIP, SE3Labs,
  Stanford_MSL, UTA_RPL) → quarantined. 0 NaN/Inf; 6 degenerate trajectories; 50 labs
  fully clean.
- **Phase B — done as a 5-lab/135-ep subset only.** Both formats verified to load on CPU.
- **Phase C — smoke-tested, not committed.** Converters validated end-to-end (pi05 DGX
  overfit passed; vla_foundry local smoke passed) but the files are uncommitted (below).

## Main remaining work

1. **Scale Phase B to the full corpus** — this is the big one.
   - Full cost: **~6–8 TB as WebDataset** (per-frame JPEG, scales with frames) or
     **~0.5–0.7 TB as LeRobot** (224² AV1). ~3–4 weeks serial on one node; CMU_Aire alone
     is ~11 days (8,580 frames/ep × 4,180 eps).
   - **Do NOT materialize on puget root** — nearly filled `/` last time. Stream straight to
     S3 or use a data volume (e.g. aws-anthony-4:/data, 1.8 TB). The exporters already
     stream-to-S3 with a disk-free abort floor; keep that.
   - Recommend **video-backed WDS** or **strided sampling of long-episode labs** → ~10×
     reduction.
   - Freeze `speed_bucket` tertile thresholds over the whole corpus first.
2. **Finish per-episode censuses** (both are cheap — read only h5 root attrs):
   - `robot_profile.robot_name` per episode → exact YAM vs non-YAM split (CMU_Aire is a
     YAM + xarm7 mix; the sampled profile alone is misleading).
   - `episode_annotations.success` coverage per lab → confirm every lab/episode is
     actually labeled (sampling said yes, never censused).
3. **Commit the Phase C converters** (see below) so training can read `oopsie_clean/`
   directly.

## Not yet committed (local on the puget checkout — need to be pulled in)

These are the trainer-side readers, smoke-tested but **uncommitted**:

- **pi05 / openpi** (`~/Documents/dsrl_openpi_flip`):
  - `yam_dataset_builder/yam_dataset_builder/convert_oopsie_to_lerobot.py`
  - configs `pi05_oopsie_smoke` + `pi05_oopsie_clean_subset` in
    `src/openpi/training/config.py` (clones of `pi05_yam`; `pi05_yam` untouched).
  - additive edits: `transforms.py` (`MetadataPrefixPrompt`), `oopsie_policy.py`.
- **vla_foundry / Qwen** (`~/Documents/vla_foundry`, branch `anthony/qwen-yam-multitask-all`):
  - `vla_foundry/data/preprocessing/robotics/converters/oopsie.py` (+ registered in
    `converters/__init__.py`, `OopsiePreprocessParams` in `preprocess_params.py`)
  - `vla_foundry/data/processor/robotics_processor.py` (`build_metadata_prefix`)
  - `params/data_params.py` (`MetadataConditioningParams`)
  - configs `config_presets/preprocessing/oopsie_metacond_smoke.yaml`,
    `config_presets/training_jobs/qwen_oopsie_metacond.yaml`,
    `config_presets/training_jobs/qwen_oopsie_clean_subset.yaml`.

## Gotchas (full detail in `TRAINING.md`)

- **LeRobot:** force `video_backend="pyav"` (this build ignores `LEROBOT_VIDEO_BACKEND`;
  torchcodec won't load); set `HF_LEROBOT_HOME` **before** `import lerobot` and pass
  `root=`.
- **WebDataset:** set `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` and pin to the vla_foundry `.venv`
  (ray relaunches workers via `uv run` → loses h5py).
- **pi05 DGX train:** `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` (80 GB OOMs at the 0.75
  default); apply the pyav sitecustomize shim (fresh `uv sync` pulls a CUDA13 torchcodec
  that fails on CUDA12 drivers). pi05 is 80 GB-only (puget GPUs OOM).
- **Disk:** never fill puget `/` — stream big outputs to S3 / a data volume.
