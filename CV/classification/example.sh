set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/../../data/cifar10}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/example}"
DEVICE="${DEVICE:-cuda}"
WANDB_PROJECT="${WANDB_PROJECT:-tcg-cifar10-example}"
EPSILON="${EPSILON:-1}"
SEED="${SEED:-1}"

COMMON_ARGS=(
  --data-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --arch cnn
  --base-opt sgd
  --target-epsilon "${EPSILON}"
  --device "${DEVICE}"
  --wandb-project "${WANDB_PROJECT}"
  --seed "${SEED}"
)

echo "[CIFAR-10] Running DP-SGD baseline"
"${PYTHON_BIN}" "${SCRIPT_DIR}/cifar10.py" \
  "${COMMON_ARGS[@]}" \
  --method baseline

echo "[CIFAR-10] Running DP-SGD + TCG"
"${PYTHON_BIN}" "${SCRIPT_DIR}/cifar10.py" \
  "${COMMON_ARGS[@]}" \
  --method tcg \
  --gate-type hard

echo "[DONE] CIFAR-10 example finished."
