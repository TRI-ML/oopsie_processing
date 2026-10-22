# Cleaned-subset manifest schema

`stage10_manifest.run()` writes one row per CLEAN episode to `manifest.jsonl` and
`manifest.parquet` (identical content; parquet JSON-encodes the nested dict columns
marked † below). Only episodes that survived Phase-A quarantine + per-episode
alignment skips, and that resolve against the embodiment registry, are emitted.

| field                     | type            | notes |
|---------------------------|-----------------|-------|
| `episode_id`              | str             | `<lab>_<h5-stem>`; stable, matches both exporters |
| `lab_id`                  | str             | donor-lab / S3-prefix name |
| `embodiment_id`           | int             | registry id (Enatic=2 matches vla_foundry) |
| `embodiment_name`         | str             | e.g. `openarm_biarm` |
| `robot_name`              | str             | from robot_profile, else registry |
| `is_biarm`                | bool            | |
| `gripper_name`            | str             | |
| `operator_name`           | str/null        | raw operator string (prefix uses first token) |
| `success`                 | float (0/1)/null| consolidated optimality label |
| `taxonomy_outcome`        | str/null        | `success` / `failure` / taxonomy string |
| `failure_category` †      | list/str/null   | taxonomy failure categories |
| `severity`                | str             | |
| `annotator`               | str/null        | |
| `label_source`            | str             | `annotation` \| `folder` \| `unknown` (never inferred) |
| `speed`                   | float           | mean \|Δaction[:valid]\| × control_freq |
| `speed_bucket`            | int (0/1/2)     | tertile over the selected clean set |
| `language_instruction`    | str             | whitespace-collapsed display form |
| `instruction_norm_key`    | str             | casefolded key for de-dup/grouping |
| `camera_roles` †          | dict            | canonical role → native camera name |
| `camera_s3_keys` †        | dict            | canonical role → absolute S3 key of the mp4 |
| `native_action_signature` †| dict           | `{motion_key,motion_dim,gripper_key,gripper_dim,valid}` |
| `T`                       | int             | episode length (frames) |
| `fps`                     | float/null      | control_freq |
| `h5_key`                  | str             | S3 key of the source h5 (read-only) |

Harmonization is keyed by embodiment (`configs/embodiments.yaml`): motion+gripper are
left-aligned into a common-18 vector, `[valid:18]` zero-padded, with `action_mask`
(a.k.a. `action_dim_valid`) marking `[:valid]`. Internally-mixed labs are harmonized
per-episode from the registry's ordered candidate slots and flagged by stage 5; the
subset restricts CMU_Aire to `grocery_packing/` (single cartesian-14 signature).
