#!/usr/bin/env bash
set -euo pipefail

CONDA="${CONDA:-/Users/diaciuc/Software/tools/miniforge3/bin/conda}"
ENV_PATH="${ENV_PATH:-/Users/diaciuc/Software/tools/miniforge3/envs/esm2}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CUDA="${CUDA:-cu121}" # set to "cpu" for CPU-only
INSTALL_FAISS="${INSTALL_FAISS:-0}"
SKIP_WEIGHTS="${SKIP_WEIGHTS:-0}"

"$CONDA" env list | grep -q "$ENV_PATH" || "$CONDA" create -y -p "$ENV_PATH" "python=${PYTHON_VERSION}" pip
source "$(dirname "$CONDA")/../etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

if [[ "$CUDA" == "cpu" ]]; then
  pip install torch torchvision torchaudio
else
  pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA}"
fi

pip install fair-esm numpy tqdm pytest

if [[ "$INSTALL_FAISS" == "1" ]]; then
  pip install faiss-cpu
fi

if [[ "$SKIP_WEIGHTS" == "0" ]]; then
  python - <<'PY'
import esm
model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
print("esm2_t36_3B_UR50D loaded", model.num_layers, model.embed_dim)
PY
fi
