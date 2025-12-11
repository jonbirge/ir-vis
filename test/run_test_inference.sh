#!/usr/bin/env bash
set -euo pipefail

# Simple runner for inference on the bundled test images.
# Assumes you're running inside the project root and a venv exists at ./venv
# and that required Python packages (torch, torchvision, etc.) are installed.

VENV_DIR="venv"
ACTIVATE="$VENV_DIR/bin/activate"

if [ -f "$ACTIVATE" ]; then
  # shellcheck disable=SC1091
  source "$ACTIVATE"
else
  echo "Warning: $ACTIVATE not found — not sourcing a virtualenv." >&2
fi

IR_IMG="test_input.png"
REF_IMG="test_ref.png"
CKPT="model.pt"
OUT_IMG="test_output.png"

echo "Running inference: IR=$IR_IMG REF=$REF_IMG CKPT=$CKPT OUT=$OUT_IMG"

python ../inference.py "$IR_IMG" "$REF_IMG" --checkpoint "$CKPT" --output "$OUT_IMG"

echo "Done. Output: $OUT_IMG"
