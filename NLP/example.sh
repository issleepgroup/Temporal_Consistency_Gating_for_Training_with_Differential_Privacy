set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results/example}"
DEVICE="${DEVICE:-cuda}"
WANDB_PROJECT="${WANDB_PROJECT:-tcg-sst2-example}"
EPSILON="${EPSILON:-1}"
CLIP_NORM="${CLIP_NORM:-0.4}"
SEED="${SEED:-42}"

COMMON_ARGS=(
  --task-name sst-2
  --model-name-or-path bert-base-uncased
  --init-mode pretrained
  --output-dir "${OUTPUT_DIR}"
  --target-epsilon "${EPSILON}"
  --num-train-epochs 3
  --per-example-max-grad-norm "${CLIP_NORM}"
  --device "${DEVICE}"
  --wandb-project "${WANDB_PROJECT}"
  --seed "${SEED}"
)

echo "[SST-2] Running DP-Adam baseline"
"${PYTHON_BIN}" "${SCRIPT_DIR}/bert.py" \
  "${COMMON_ARGS[@]}" \
  --method baseline

echo "[SST-2] Running DP-Adam + TCG"
"${PYTHON_BIN}" "${SCRIPT_DIR}/bert.py" \
  "${COMMON_ARGS[@]}" \
  --method tcg \
  --gate-type soft \
  --gate-tau 0.05

echo "[DONE] SST-2 example finished."
