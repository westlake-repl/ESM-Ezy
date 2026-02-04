# AGENTS.md

## Goal
Make the model accept ESM2 features from either:
1) fair-esm (on-the-fly inference) using `esm2_t36_3B_UR50D`, or
2) precomputed embeddings (e.g., ESM Atlas-style exports),
while keeping downstream architecture stable via a projection to `d_model`.

Must run on CPU or GPU (CUDA and/or Apple MPS). Must use the conda env at:
`/Users/diaciuc/Software/tools/miniforge3/envs/esm2`

## Environment (Codex must use this env)
### 0) Create/activate the env (empty env is OK)
```bash
CONDA="/Users/diaciuc/Software/tools/miniforge3/bin/conda"
ENV="/Users/diaciuc/Software/tools/miniforge3/envs/esm2"

# create if missing
"$CONDA" env list | rg -q "$ENV" || "$CONDA" create -y -p "$ENV" python=3.10 pip

# activate
source "/Users/diaciuc/Software/tools/miniforge3/etc/profile.d/conda.sh"
conda activate "$ENV"

python -V
pip -V
