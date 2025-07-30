# ESM-Ezy

## Dataset and checkpoint

To get dataset and model checkpoint, please refer to [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15258919.svg)](https://doi.org/10.5281/zenodo.15258919).

Download the `data.zip` file and extract it to the `data` directory.

Download the `ckpt.zip` file and extract it to the `ckpt` directory.

### About the `train_positive.fa` and `train_negative.fa` files

We apologize that the `train_positive.fa` (referred to as the original training positive file) and `train_negative.fa` (referred to as the original training negative file) included in `data.zip` do not exactly match the manuscript description. The original training positive file, which initially contained **117** entries, underwent incomplete data duplication. The revised `train_positive.fa` located in root directory of [Zenodo](https://doi.org/10.5281/zenodo.15258919) now includes deduplicated **117** entries. Similarly, the `train_negative.fa` has been updated. It is important to note, however, that since the negative samples are **randomly sampled from `train_negative.fa` during training** (which is sufficiently large relative to `train_positive.fa`), minor changes to the latter have minimal impact on the training process. So even if you have downloaded the past version of `train_positive.fa` and `train_negative.fa`, the training process should still work fine.

## Training

To train ESM-Ezy, follow the steps below:

1. Clone the repository:

```
git clone https://github.com/westlake-repl/ESM-Ezy.git
```

2. Install the required packages:

```
conda env create -f environment.yml
```

3. Download the pre-trained ESM-1b model:

```
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt -O ckpt/esm1b_t33_650M_UR50S.pt
wget https://dl.fbaipublicfiles.com/fair-esm/regression/esm1b_t33_650M_UR50S-contact-regression.pt -O ckpt/esm1b_t33_650M_UR50S-contact-regression.pt
```

4. Train ESM-Ezy:

```
python scripts/train.py --train_positive_data data/train/train_positive.fa --train_negative_data data/train/train_negative.fa --test_positive_data data/train/test_positive.fa --test_negative_data data/train/test_negative.fa --model_path ckpt/esm1b_t33_650M_UR50S.pt

We also add early stopping to determine the training process is ready, you can try with:
python scripts/train_earlystop.py --train_positive_data data/train/train_positive.fa --train_negative_data data/train/train_negative.fa --test_positive_data data/train/test_positive.fa --test_negative_data data/train/test_negative.fa --model_path ckpt/esm1b_t33_650M_UR50S.pt --patience 10

```

## inference

1. inference from uniref50 database:

```
python scripts/inference.py --model_path ckpt/esm1b_t33_650M_UR50S.pt --checkpoint_path ckpt/model_laccase.pkl --inference_data data/inference/uniref50.fasta  --output_path data/retrieval
```

## Search

1. load the trained ESM-Ezy model and inference on the candidate sequences:

```
python scripts/retrieval.py --model_path ckpt/esm1b_t33_650M_UR50S.pt --checkpoint_path ckpt/model_laccase.pkl --candidate_data data/retrieval/candidate.fa --seed_data data/retrieval/fitness.fa  --output_path data/retrieval
```
