# ESM-Ezy

## Dataset and checkpoint

To get dataset and model checkpoint, please refer to [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12776629.svg)](https://doi.org/10.5281/zenodo.12776629).

Download the `data.zip` file and extract it to the `data` directory.

Download the `ckpt.zip` file and extract it to the `ckpt` directory.

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
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S-contact-regression.pt -O ckpt/esm1b_t33_650M_UR50S-contact-regression.pt
```

4. Train ESM-Ezy:

```
python scripts/train.py --train_positive_data data/train/train_positive.fa --train_negative_data data/train/train_negative.fa --test_positive_data data/train/test_positive.fa --test_negative_data data/train/test_negative.fa --model_path ckpt/esm1b_t33_650M_UR50S.pt
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
