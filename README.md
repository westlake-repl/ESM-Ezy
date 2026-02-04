# ESM2-Ezy

ESM2-based pipeline that supports **fair-esm** embeddings or **precomputed** embeddings based on ESM-Ezy pipeline.

![CI](https://img.shields.io/github/actions/workflow/status/paleblueillud/esm2_ezy/ci.yml?branch=main)
![License](https://img.shields.io/github/license/paleblueillud/esm2_ezy)
![Python](https://img.shields.io/badge/python-3.10-blue)

## Table of contents
- [Quickstart](#quickstart)
- [Installation](#installation)
  - [CPU (portable)](#cpu-portable)
  - [CUDA (Linux cluster)](#cuda-linux-cluster)
  - [Apple Silicon (MPS)](#apple-silicon-mps)
- [Running](#running)
  - [fair-esm embeddings (on-the-fly)](#fair-esm-embeddings-on-the-fly)
  - [Precomputed embeddings](#precomputed-embeddings)
- [Caching and offline clusters](#caching-and-offline-clusters)
- [Tests](#tests)
- [Reproducibility notes](#reproducibility-notes)

## Quickstart
```bash
mamba env create -f environment.yml -n esm2_ezy
conda activate esm2_ezy
python -c "import torch, esm, numpy; print('ok')"
python scripts/inference.py \
  --inference_data data/petase/pazy.fasta \
  --output_path /tmp/esm2_ezy_infer \
  --embedding_source fair_esm2 \
  --esm_checkpoint esm2_t6_8M_UR50D \
  --device auto
pytest -q
```

## Installation

### CPU (portable)
Prefer `mamba` (faster), or use `conda` if needed.

```bash
mamba env create -f environment.yml -n esm2_ezy
conda activate esm2_ezy
```

### CUDA (Linux cluster)
Create the base env, then upgrade PyTorch with CUDA support.

```bash
mamba env create -f environment.yml -n esm2_ezy
conda activate esm2_ezy

# Choose a CUDA version compatible with your cluster/driver.
conda install -n esm2_ezy pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Apple Silicon (MPS)
```bash
mamba env create -f environment.yml -n esm2_ezy
conda activate esm2_ezy
python - <<'PY'
import torch
print("mps available:", torch.backends.mps.is_available())
PY
```

## Running

### Training
Training expects FASTA files for positive/negative splits.

```bash
python scripts/train.py \
  --train_positive_data data/petase/positive_train.fasta \
  --train_negative_data data/petase/negative_train.fasta \
  --test_positive_data data/petase/positive_test.fasta \
  --test_negative_data data/petase/negative_test.fasta \
  --embedding_source fair_esm2 \
  --esm_checkpoint esm2_t36_3B_UR50D \
  --device auto \
  --precision fp32 \
  --batch_size 16 \
  --epoch 100 \
  --patience 10
```

If you use precomputed embeddings, pass `--embedding_source precomputed` and `--precomputed_embeddings_dir`.

### fair-esm embeddings (on-the-fly)
Requires sequences (FASTA). ESM2 weights will download on first use.

```bash
python scripts/inference.py \
  --inference_data data/petase/pazy.fasta \
  --output_path /tmp/esm2_ezy_infer \
  --embedding_source fair_esm2 \
  --esm_checkpoint esm2_t36_3B_UR50D \
  --device auto \
  --precision fp32
```

### Precomputed embeddings
Requires IDs that map to embedding files. Supported formats: `.npy`, `.npz`, `.pt`.

Directory with one file per ID:
```bash
python scripts/inference.py \
  --inference_data data/petase/pazy.fasta \
  --output_path /tmp/esm2_ezy_infer \
  --embedding_source precomputed \
  --precomputed_embeddings_dir /path/to/embeddings \
  --precomputed_format auto \
  --precomputed_granularity per_sequence
```

### Search / Retrieval
Use the candidate FASTA and a seed FASTA. FAISS is optional; disable it via `ESM_EZY_USE_FAISS=0`.

```bash
ESM_EZY_USE_FAISS=0 python scripts/retrieval.py \
  --candidate_data data/petase/pazy.fasta \
  --seed_data data/petase/pazy_search_1.fasta \
  --output_path results/inference \
  --embedding_source fair_esm2 \
  --esm_checkpoint esm2_t36_3B_UR50D \
  --device auto \
  --precision fp32
```

Single `.npz` container (keys are IDs; no FASTA required):
```bash
python scripts/inference.py \
  --output_path results/search \
  --embedding_source precomputed \
  --precomputed_embeddings_dir /path/to/embeddings.npz \
  --precomputed_format npz \
  --precomputed_granularity per_sequence
```

## Caching and offline clusters
Set `TORCH_HOME` to a shared filesystem to reuse cached weights:

```bash
export TORCH_HOME=/shared/torch-cache
python -c "import esm; esm.pretrained.esm2_t36_3B_UR50D(); print('ok')"
```

Download weights on a login node, then run jobs with the same `TORCH_HOME`.

## Tests
```bash
pytest -q
```

## Reproducibility notes
```bash
conda env create -f environment.yml -n esm2_ezy
conda env export --no-builds > env.lock.yml
```

If you need fully locked environments, consider `conda-lock` as a follow-up.
