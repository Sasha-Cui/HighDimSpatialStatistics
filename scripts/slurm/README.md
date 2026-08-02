# SLURM Scripts (Updated)

These scripts are updated to work with the new repository layout.

- Scripts mapped to the new pipeline use `python -m scripts.pipeline.<module>`.
- Scripts without a pipeline equivalent invoke the converted legacy snapshots via
  `python -m scripts.pipeline.legacy_runner`.

Legacy originals are preserved in `scripts/slurm/legacy/`.

## Logging

Set `HDS_LOG_DIR` to control where logs (including GPU logs) are written.

## Device

Set `HDS_DEVICE` to `cpu` or `cuda` for pipeline scripts that accept `--device`.

## Legacy Runner

Scripts like `batch.sh` and `myjob.sh` expect `HDS_LEGACY_SCRIPT` to be set to a legacy script path.

The legacy runner does not make historical notebooks reproducible: many still
require unavailable packages, hard-coded data, and obsolete return shapes. These
jobs are provenance aids only. The cross fitter is also under mathematical audit
and must not be used for paper results. The smoothing jobs save operator-aware
groups, but no production smoothing-aware fitter exists yet.
