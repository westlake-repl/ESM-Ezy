# ESM-Ezy

## Training

To train ESM-Ezy, follow the steps below:

1. Clone the repository:

```
git clone https://github.com/westlake-repl/ESM-Ezy.git
```

2. Install the required packages:

```
pip install -r requirements.txt
```

3. Download the pre-trained ESM-1b model:

```
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt -O ckpt/esm1b_t33_650M_UR50S.pt
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S-contact-regression.pt -O ckpt/esm1b_t33_650M_UR50S-contact-regression.pt
```

4. Train ESM-Ezy:

```
python train.py --positive_data data/positive_data.pkl --negative_data data/negative_data.pkl
```

## Search

1. load the trained ESM-Ezy model and inference on the candidate sequences:

```
python inference.py --model_path ckpt/model_laccase.pkl --candidate_data data/candidate_data.pkl --seed_data data/seed_data.pkl --output_path output/
```
