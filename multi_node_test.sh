#!/usr/bin/env bash
set -euo pipefail

USER_NAME="${USER}"
REPO="/shared/${USER_NAME}/miracle/pv_forecast_30d"
IMG="/shared/${USER_NAME}/miracle/containers/tft_env_v1.sif"
LOG_DIR="/shared/${USER_NAME}/miracle/logs"
RUN_ROOT="/shared/${USER_NAME}/miracle/experiments/tft/runs/germany/v1_0"

TRAIN_SRC="${REPO}/data/processed/pretraining/germany/global/tft_inputs/regional_train_tft_full.parquet"
VAL_SRC="${REPO}/data/processed/pretraining/germany/global/tft_inputs/regional_val_tft_full.parquet"

NODES=("dbfz-hpc23-gnode1" "dbfz-hpc23-gnode2" "dbfz-hpc23-gnode3" "dbfz-hpc23-gnode4")

MAX_EPOCHS=1
BATCH_SIZE=1024
NUM_WORKERS=0
ENC_LAG=96
PRECISION="bf16-mixed"
ENABLE_AMP=1
THREADS=4

mkdir -p "${LOG_DIR}"

[[ -d "${REPO}" ]] || { echo "[ERROR] REPO not found: ${REPO}"; exit 1; }
[[ -f "${IMG}"  ]] || { echo "[ERROR] IMG not found:  ${IMG}"; exit 1; }
[[ -f "${TRAIN_SRC}" ]] || { echo "[ERROR] Train parquet not found: ${TRAIN_SRC}"; exit 1; }
[[ -f "${VAL_SRC}"   ]] || { echo "[ERROR] Val parquet not found:   ${VAL_SRC}"; exit 1; }

# Decide AMP argument once, in wrapper, as a literal
AMP_ARG=""
if [[ "${ENABLE_AMP}" = "1" ]]; then
  AMP_ARG="--enable_amp"
fi

echo "Submitting node tests..."
echo "Nodes: ${NODES[*]}"
echo "AMP_ARG: ${AMP_ARG:-<none>}"
echo "Logs: ${LOG_DIR}"
echo

for node in "${NODES[@]}"; do
  case "${node}" in
    dbfz-hpc23-gnode1|dbfz-hpc23-gnode2) PARTITION="gpua100" ;;
    dbfz-hpc23-gnode3|dbfz-hpc23-gnode4) PARTITION="gpuh100" ;;
    *) PARTITION="gpu" ;;
  esac

  job_name="gpu_test_${node}"
  echo "Submitting ${job_name} on ${node} (${PARTITION})"

  sbatch \
    --job-name="${job_name}" \
    --partition="${PARTITION}" \
    --nodelist="${node}" \
    --gres=gpu:1 \
    --cpus-per-task=16 \
    --mem=250G \
    --time=00:30:00 \
    --output="${LOG_DIR}/test_%N_%j.out" \
    --error="${LOG_DIR}/test_%N_%j.err" \
<<EOF
#!/usr/bin/env bash
set -euo pipefail

echo "=== NODE TEST START ==="
echo "UTC: \$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "Host: \$(hostname)"
echo "SLURM_JOB_ID: \${SLURM_JOB_ID:-}"
echo "Partition: ${PARTITION}"
echo

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load singularity >/dev/null 2>&1 || true
fi

command -v singularity >/dev/null 2>&1 || { echo "[ERROR] singularity not found in PATH"; exit 3; }

export SINGULARITY_BINDPATH="/shared,/tmp"
export OMP_NUM_THREADS=${THREADS}
export MKL_NUM_THREADS=${THREADS}
export OPENBLAS_NUM_THREADS=${THREADS}
export NUMEXPR_NUM_THREADS=${THREADS}
export CUDA_VISIBLE_DEVICES=0

LOCAL_DIR="/tmp/${USER_NAME}/\${SLURM_JOB_ID}"
mkdir -p "\${LOCAL_DIR}"
trap 'echo "=== CLEANUP ==="; rm -rf "\${LOCAL_DIR}"' EXIT

cp "${TRAIN_SRC}" "\${LOCAL_DIR}/train.parquet"
cp "${VAL_SRC}"   "\${LOCAL_DIR}/val.parquet"

echo "Local train: \${LOCAL_DIR}/train.parquet"
echo "Local val:   \${LOCAL_DIR}/val.parquet"
echo

# NOTE: keep singularity call in the cluster-accepted pattern
singularity exec --nv "${IMG}" bash -lc "
  set -euo pipefail
  cd '${REPO}'
  export PYTHONPATH='${REPO}':\${PYTHONPATH:-}

  echo 'Inside container:'
  python3 -V
  nvidia-smi -L
  echo

  echo '--- GPU POWER / CLOCKS (start) ---'
  nvidia-smi --query-gpu=name,pstate,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,temperature.gpu --format=csv
  echo

  python3 -u -m src.training.train_tft_v1 \
    --train_parquet '\${LOCAL_DIR}/train.parquet' \
    --val_parquet   '\${LOCAL_DIR}/val.parquet' \
    --run_root      '${RUN_ROOT}' \
    --enc_lag ${ENC_LAG} \
    --max_epochs ${MAX_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --lr 3e-4 \
    --weight_decay 1e-4 \
    --patience 1 \
    --min_delta 1e-5 \
    --log_every_n_steps 0 \
    --precision '${PRECISION}' \
    ${AMP_ARG}

  echo
  echo '--- GPU POWER / CLOCKS (end) ---'
  nvidia-smi --query-gpu=power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu --format=csv
"

echo "=== NODE TEST END ==="
echo "UTC: \$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
EOF

done

echo
echo "Monitor:"
echo "  squeue -u \$USER -o \"%.18i %.9P %.18j %.2t %.10M %.20N %.30R\""
echo "Errors:"
echo "  ls -lt ${LOG_DIR}/test_*.err | head"
