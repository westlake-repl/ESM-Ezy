import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.esm_model import LaccaseModel
from dataset.retrieval_dataset import RetrievalDataset
from torch.utils.data import DataLoader, DistributedSampler
import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
import faiss
import mkl
mkl.get_max_threads()

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument('-ckpt', '--checkpoint_path', type=str, default=None)
    parser.add_argument('-q', '--query_path', type=str, required=True)
    parser.add_argument('-repr', '--repr_path', type=str, required=True)