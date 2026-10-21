# oopsie_processing

Tooling and plan for cleaning and standardizing the **oopsie** multi-institution
robot-manipulation dataset into a training-ready, conditioning-ready corpus —
without destroying the raw originals.

- **Raw dataset:** `s3://tri-ml-datasets-uw2/anthonyliang/oopsie/` (us-west-2) —
  1.89 TB, 490,288 objects, 84 top-level prefixes, 65 real contributing labs.
- **Schema:** `oopsiedata_format_v1` — one small `.h5` trajectory per episode
  (lowdim only; actions + proprio) plus one `.mp4` per camera (images are external;
  the h5 holds only path strings).
- **Audit + viewer (what motivates this cleanup):**
  <https://aliang8.github.io/reports/oopsie/> (password `oopsie`).

The full cleaning plan — scope, the eleven cleaning stages, the non-destructive
output contract, and phasing — is in **[PLAN.md](PLAN.md)**.

## Principle

**Non-destructive.** Nothing in the raw `oopsie/` prefix is deleted or mutated.
Every stage reads the raw corpus and writes a cleaned, versioned derivative to a
new prefix (`oopsie_clean/`, schema `oopsiedata_format_v2`) alongside a per-episode
metadata manifest. The raw corpus remains the source of truth.

## Status

Planning. See PLAN.md for the staged implementation. Code lands under `src/` as each
stage is built; each stage is independently runnable and emits a report.
