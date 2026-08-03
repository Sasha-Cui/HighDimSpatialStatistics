#!/bin/bash
set -euo pipefail

REPO_ROOT=/nfs/roberts/project/pi_btk22/zc362/HighDimSpatialStatistics
HDS_PYTHON=${HDS_PYTHON:-/nfs/roberts/project/pi_btk22/zc362/environments/current/kt-main/bin/python}
SBATCH=/opt/slurm/current/bin/sbatch
MANIFEST=${1:-$REPO_ROOT/configs/smoothing_bias/shakedown_20260802.json}

cd "$REPO_ROOT"
mkdir -p outputs/slurm

TASK_COUNT=$(
  "$HDS_PYTHON" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["configurations"]))' "$MANIFEST"
)
if (( TASK_COUNT < 1 || TASK_COUNT > 180 )); then
  echo "Refusing to submit $TASK_COUNT array elements; required range is 1..180" >&2
  exit 2
fi

ARRAY_JOB=$(
  "$SBATCH" --parsable --array="0-$((TASK_COUNT - 1))%8" \
    scripts/slurm/smoothing_bias_array.sbatch "$MANIFEST"
)
REDUCER_JOB=$(
  "$SBATCH" --parsable --dependency="afterany:$ARRAY_JOB" \
    scripts/slurm/smoothing_bias_reduce.sbatch "$MANIFEST"
)

echo "Submitted one $TASK_COUNT-element array: $ARRAY_JOB"
echo "Submitted one afterany reducer: $REDUCER_JOB"
echo "Total sbatch submissions: 2 (policy maximum: 180 per hour)"
