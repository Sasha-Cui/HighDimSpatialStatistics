# SLURM Scripts (Updated)

These scripts are updated to work with the new repository layout.

- Scripts mapped to the new pipeline call `scripts/pipeline/*.py`.
- Scripts without a pipeline equivalent run the converted legacy scripts via `scripts/pipeline/legacy_runner.py`.

Legacy originals are preserved in `scripts/slurm/legacy/`.

## Logging

Set `HDS_LOG_DIR` to control where logs (including GPU logs) are written.

## Device

Set `HDS_DEVICE` to `cpu` or `cuda` for pipeline scripts that accept `--device`.

## Legacy Runner

Scripts like `batch.sh` and `myjob.sh` expect `HDS_LEGACY_SCRIPT` to be set to a legacy script path.
