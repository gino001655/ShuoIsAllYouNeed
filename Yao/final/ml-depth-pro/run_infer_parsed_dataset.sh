#!/usr/bin/env bash
set -euo pipefail

# Run layer-order inference over the whole parsed_dataset.
#
# Defaults:
# - DATA_DIR: ../parsed_dataset (relative to ml-depth-pro/)
# - OUT_DIR:  ./infer_outputs/parsed_dataset_pred
# - CKPT_PATH: auto-pick newest checkpoints/layer_order/experiment_*/checkpoint_best.pt
#
# You can override by exporting env vars before running:
#   CKPT_PATH=... DATA_DIR=... OUT_DIR=... ./run_infer_parsed_dataset.sh
#
# Or pass flags through AFTER `--`:
#   ./run_infer_parsed_dataset.sh -- --splits test --use-amp --skip-existing

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="${DATA_DIR:-../parsed_dataset}"
OUT_DIR="${OUT_DIR:-./infer_outputs/parsed_dataset_pred}"

if [[ -z "${CKPT_PATH:-}" ]]; then
  # Pick the newest checkpoint_best.pt under the default training output tree.
  CKPT_PATH="$(ls -1dt ./checkpoints/layer_order/experiment_*/checkpoint_best.pt 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${CKPT_PATH:-}" ]]; then
  echo "[ERROR] CKPT_PATH is not set and no checkpoint_best.pt was found under ./checkpoints/layer_order/experiment_*/"
  echo "        Please set CKPT_PATH explicitly:"
  echo "        CKPT_PATH=/path/to/checkpoint_best.pt ./run_infer_parsed_dataset.sh"
  exit 1
fi

EXTRA_ARGS=()
if [[ "${1:-}" == "--" ]]; then
  shift
  EXTRA_ARGS=("$@")
fi

echo "[InferSH] Using:"
echo "  - DATA_DIR=${DATA_DIR}"
echo "  - OUT_DIR=${OUT_DIR}"
echo "  - CKPT_PATH=${CKPT_PATH}"
echo "  - EXTRA_ARGS=${EXTRA_ARGS[*]:-(none)}"

python custom_layer_order_infer_parsed_dataset.py \
  --data-dir "${DATA_DIR}" \
  --ckpt-path "${CKPT_PATH}" \
  --out-dir "${OUT_DIR}" \
  "${EXTRA_ARGS[@]}"





