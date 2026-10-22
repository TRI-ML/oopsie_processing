# oopsie — Phase A quality checks

This is the read-only inventory of the raw oopsie corpus
(`s3://tri-ml-datasets-uw2/anthonyliang/oopsie/`). Nothing under that prefix was
modified or deleted. Every number below comes from running the Phase-A stages in
this repo (`python run.py phaseA`) against the precomputed object listing and a
bounded, evenly-spaced sample of the actual h5 and mp4 files. The machine-readable
output for each stage lives in `reports/` (gitignored) and is mirrored for Phase B.

Sampling: up to 150 h5 per lab for schema and data-quality (8,631 episodes read),
25 episodes per lab for alignment (1,502 episodes, each probing its cameras).
mp4 frame counts and fps are read from the container header over a presigned URL
(no full download); this was spot-checked against exact `ffprobe -count_frames`
and matched to the frame on every case tried.

## Scope: what is a lab and what is not

The corpus has 84 top-level prefixes in the object listing. Only 61 are real
contributing labs. The audit's `per_lab` map lists 65 keys; four of those are
infrastructure, not labs (`SOAR_origin`, `AutoEval_origin`,
`oopsiedata-converted-v0.1`, `roboarena_upload`), which reconciles 65 − 4 = 61.
The listing-based classification and the audit-derived count agree exactly.

The 61 real labs hold 82,182 h5 episodes and about 1,837 GB. The remaining 23
prefixes carry roughly 51 GB and are all excluded from the clean set:

| Category | Count | Prefixes | Size |
|---|---|---|---|
| real lab (include) | 61 | — | 1,837.4 GB |
| source / origin | 2 | SOAR_origin, AutoEval_origin | 18.1 GB |
| duplicate | 2 | oopsiedata-converted-v0.1, oopsie-face-raw-archive-2026-08-12 | 14.3 GB |
| preview | 2 | WebVisualizer, Visualizer_Previews | 0.1 GB |
| staging | 1 | roboarena_upload | 18.9 GB |
| empty placeholder | 15 | CMU_RCHI, DHBWR_MS, Duke_DexLab, MIT_AT, NUS_JD, NW_RZ, NYU_LR, NomadicAI, PrismaX_BW, Sejong_RCV, Stanford_ILIAD, TU_GC, UMD_RL, UPenn_JIRL, qa_out | 0 |
| HF cache | 1 | `.cache/` | — |

The authoritative include/exclude list is `reports/prefix_taxonomy.json`.

## Duplication

`oopsiedata-converted-v0.1` is confirmed redundant. The h5 inside it carry
`lab_id` attributes pointing back to real labs (in the sample, 59/60 to
Berkeley_RAIL and 1/60 to AI2_MOLMOACT2), and one sampled file is byte-identical
to the AI2_MOLMOACT2 original. It is a re-converted copy of real-lab episodes, so
the bytes are mostly not identical but the provenance is unambiguous.

`oopsie-face-raw-archive-2026-08-12` is an archive of 240 mp4, every one of which
shares a filename with an mp4 in a real lab (204 in UofT_TISL, the rest spread
across Stanford_REAL, UW_REAL, Princeton_PRISM, AI2_MOLMOACT2, UPenn_PAL). The
bytes are not identical to the originals (0 of 30 header-compared), so these are
re-encoded (face-processed) copies rather than exact duplicates. Either way it is
an archive, not a lab, and is excluded.

A cross-lab byte-identical scan over 1,502 h5 (25 per real lab) found zero
episodes shared byte-for-byte across different labs. Details in `reports/dedup.json`.

## Cruft and defects (quarantine, never deleted)

The quarantine list is `reports/quarantine.jsonl`: 15,554 paths, dominated by
cruft taken straight from the listing.

- `*.hidden` shadow copies: 15,498, all in USP_MB. (The audit called these
  `*.ext.hidden`; the actual suffix is `.hidden` appended to the real filename,
  e.g. `...h5.hidden`, `..._wrist.mp4.hidden`. Excluding them roughly halves that
  lab's object count.)
- `.h5.bak`: 50, all in UW_REAL.
- `.zip`: 3. `.DS_Store`: 2.
- Undersized mp4 container: 1 — CMU_Auton's 261-byte zero-content mp4, exactly the
  defect the audit called out. A listing-wide scan found no other mp4 under the
  1 KB threshold.

Bounded defect probing confirmed the data is otherwise intact: 1,206 mp4 probed
(0 unreadable or zero-frame) and 2,402 h5 opened (0 truncated or empty).

## Schema validation (oopsiedata_format_v1)

Overall 8,601 of 8,631 sampled episodes pass = 99.65%, with zero read errors.
60 of 61 labs validate at 100% in the sample: required root attrs present,
`robot_profile` parses as JSON, `robot_states` populated, and all
`observations/video_paths` pointers resolve to an mp4 that exists in the bucket.

The one failure is **Stanford_REAL at 43% (30 of 53 episodes)**. These are its
`cup_on_saucer` tasks, whose video pointers drop the task-name prefix — the h5
says `indomain_zarr_episode_000000_aria.mp4` while the file on S3 is
`cup_on_saucer_indomain_zarr_episode_000000_aria.mp4`. The videos exist; the
pointer strings are malformed. This is recoverable in Phase B by repairing the
prefix, not a case of missing footage. Per-lab records are in
`reports/schema_validation.json`.

## Temporal alignment (mp4 frame count == T, fps == control_freq)

This is the most significant finding, and it departs from the audit. The audit
spot-checked 18 labs and reported perfect alignment. Checking all 61 labs
(1,502 episodes), 1,372 are fully aligned = 91.3% of episodes, 93.0% of cameras.
Ten labs contain genuinely misaligned episodes:

| Lab | episodes mismatched / sampled | nature |
|---|---|---|
| INRIA_SC | 21/25 | video shorter than trajectory (e.g. 883 frames vs T=1079) |
| SE3Labs | 24/25 | video longer than trajectory (389 vs 274) |
| McGill_DM_HL | 18/25 | video ~2x trajectory length |
| NUS_CLeAR | 10/25 | video ~2.25x trajectory length |
| Araya | 9/25 | video longer (102 vs 81) |
| UTA_RPL | 9/25 | video slightly longer (154 vs 147) |
| PSU_FMC | 5/25 | video ~0.57x trajectory length |
| Stanford_MSL | 5/25 | video slightly longer |
| RIKEN_AIP | 4/25 | video longer (900 vs 800) |
| Harvard_EML | 11/25 | frame count matches T but declared control_freq (100 Hz) disagrees with actual video fps (10 Hz) |

The frame-count gaps are large, consistent across all cameras of an episode, and
confirmed by exact frame counting, so they are real: in these labs an mp4 frame
does not line up one-to-one with a trajectory step, and Harvard_EML's
`control_freq` metadata is simply wrong relative to its video. None of these are
probe artifacts. The flag is per-episode, not per-lab: several of these labs
(PSU_FMC, RIKEN_AIP, Stanford_MSL) are mostly aligned, so Phase B should
quarantine the mismatched episodes individually rather than drop whole labs. Full
per-lab, per-camera data is in `reports/alignment.json`.

## Data quality

Across the 8,631-episode sample (`reports/data_quality.json`):

- No NaN or Inf in any `actions` or `robot_states` array. Zero, corpus-wide.
- 6 degenerate (fully constant) action trajectories: NUS_CLeAR 3, Stanford_MSL 2,
  Stanford_REAL 1. Negligible in absolute terms but worth dropping in Phase B.
- No duration outliers in the sample under the 0.5 s / 600 s thresholds; the
  precomputed duration sweep put the 61 real labs at about 770 hours total, mean
  episode ~31 s.
- Instruction coverage is complete: `language_instruction` is present in all 61
  labs. Only two labs are single-instruction (Stanford_Iris, TUD_JP), matching the
  audit. Counts reused from `instruction_counts.json`.
- Action spaces are heterogeneous, as expected. 26 labs show more than one action
  dimensionality within their own sample, several because they mix robots
  internally — CMU_Aire (dual_ufactory_xarm7 and i2rt_yam_bimanual, control_freq
  14–60 Hz), UofT_RVL (flexiv and franka), Toloka (franka and a bimanual trossen),
  Princeton_PRPL. Recorded action value ranges vary by embodiment (e.g. CMU_Aire
  reaches ~846 in some joint/gripper channel, Toloka ~100); these are flagged for
  inspection during Phase-B harmonization, not treated as corruption.

## Is this trainable as-is?

For the big labs, yes, with the quarantine applied. The largest contributors —
CMU_Aire (4,180 h5, 542 GB, ~151 h, ~20% of the corpus), WUSTL_MIG, USP_MB (after
dropping its 15.5k shadow copies), Princeton_PRPL, Armnet, KAIST_KL,
Stanford_SIALab, Charles_UFAL, UofT_RVL, Toloka — all pass schema at 100% and show
clean alignment in the sample. These are usable now.

Three things must be handled before training, none of them blockers:

1. Exclude the 23 non-lab prefixes and the 15,554 quarantined cruft/defect paths.
2. Quarantine or repair the misaligned episodes in the ten alignment-flagged labs.
3. Repair Stanford_REAL's malformed `cup_on_saucer` video pointers (the footage is
   present).

Harmonizing the heterogeneous action spaces and consolidating labels is Phase B.

### Clean vs. flagged (all 61 real labs)

- **Clean (schema 100% + alignment clean in sample): 50 labs** — AI2_MOLMOACT2,
  Amazon-MobileManipulationTeam, Armnet, BONN_MB, Berkeley_HybridRobotics,
  Berkeley_RAIL, Brown_ST, CMU_Aire, CMU_Auton, CMU_Intent, CMU_RPAD, Charles_UFAL,
  ETH_RK, Enact, Enatic_OpenArm, GT_PAIR, GT_RL2, GT_YC, KAIST_JS, KAIST_KL,
  KIT_ALR, MIT_CLEAR, MIT_Improbable, NYU_AI4CE, NYU_Clark, Northstar,
  Princeton_IRoM, Princeton_PRISM, Princeton_PRPL, Princeton_VLL, Stanford_FFL,
  Stanford_Iris, Stanford_JW, Stanford_SIALab, TUD_JP, Toloka, UCSD_ARCLAB,
  UIUC_DASLAB, UIUC_Reid, UNC_MD, UPenn_PAL, USC_Lira, USP_MB, UT_RobIn, UW_REAL,
  UdeM_REAL, UofT_RVL, UofT_TISL, VUA_KL, WUSTL_MIG.
- **Alignment-flagged (contains misaligned episodes; quarantine per-episode): 10** —
  Araya, Harvard_EML, INRIA_SC, McGill_DM_HL, NUS_CLeAR, PSU_FMC, RIKEN_AIP,
  SE3Labs, Stanford_MSL, UTA_RPL.
- **Schema-flagged (malformed but recoverable video pointers): 1** — Stanford_REAL.
