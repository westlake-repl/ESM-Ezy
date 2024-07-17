import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.esm_model import LaccaseModel
from dataset import FastaDataset
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
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--inference_data', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    args = parse_args()
    model_path = args.model_path
    checkpoint_path = args.checkpoint_path
    inference_data = args.inference_data
    output_path = args.output_path
    
    # load model
    print("Loading model...")
    model = LaccaseModel.from_pretrained(model_path, state_dict_path=checkpoint_path)
    model = model.to(device)

    # data
    print("Reading candidate data...")
    inference_dataset = FastaDataset(inference_data)
    inference_dataloader = DataLoader(inference_dataset, batch_size=64, shuffle=True,
                                    collate_fn=inference_dataset.collate_fn, drop_last=False, pin_memory=True)
    
    inference_list = []
    with torch.no_grad():
        for content in tqdm(inference_dataloader, total=len(inference_dataloader)):
            last_result = model(content)
            mask = last_result[:, 1] > last_result[:, 0]
            inference_list.extend([c for m, c in zip(mask, content) if m])

    with open(os.path.join(output_path, "candidate.fa"), "w") as f:
        for c in inference_list:
            f.write(f">{c[0]}\n{c[1]}\n")