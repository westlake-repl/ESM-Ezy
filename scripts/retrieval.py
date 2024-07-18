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
    parser.add_argument('--candidate_data', type=str)
    parser.add_argument('--seed_data', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    args = parse_args()
    model_path = args.model_path
    checkpoint_path = args.checkpoint_path
    candidate_data = args.candidate_data
    seed_data = args.seed_data
    output_path = args.output_path
    
    # load model
    print("Loading model...")
    model = LaccaseModel.from_pretrained(model_path, state_dict_path=checkpoint_path)
    model = model.to(device)
    print(model.device)

    # data
    print("Reading candidate data...")
    candidate_dataset = FastaDataset(candidate_data)
    candidate_dataloader = DataLoader(candidate_dataset, batch_size=64, shuffle=True,
                                    collate_fn=candidate_dataset.collate_fn, drop_last=False, pin_memory=True)
    
    seed_dataset = FastaDataset(seed_data)
    seed_dataloader = DataLoader(seed_dataset, batch_size=1, shuffle=True,
                                    collate_fn=seed_dataset.collate_fn, drop_last=False, pin_memory=True)
    
    candidate_info_list = []
    with torch.no_grad():
        for j, item in tqdm(enumerate(candidate_dataloader), total=len(candidate_dataloader)):
            out_result, last_repr = model(item, return_repr=True)
            for i, r in zip(item, last_repr.cpu().numpy()):
                candidate_info_list.append((i, r))
    candidate_repr = np.stack([r for item, r in candidate_info_list], axis=0)
    print(candidate_repr.shape)


    # faiss index
    index = faiss.IndexFlatL2(1280)
    index.add(candidate_repr)
    result_list = []
    with torch.no_grad():
        for j, item in tqdm(enumerate(seed_dataloader), total=len(seed_dataloader)):
            out_result, last_repr = model(item, return_repr=True)
            D, I = index.search(last_repr.unsqueeze(0).cpu().numpy(), k=10)
            for i, distance in zip(I[0], D[0]):
                res_tuple = (item, candidate_info_list[i], distance)
                if res_tuple not in result_list:
                    result_list.append(res_tuple)

    with open(os.path.join(output_path, "results.csv"), "w") as f:
        f.write("seed_id,candidate_id,candidate_sequence,distance\n")
        for res in result_list:
            seed_info, candidate_info, distance = res
            f.write(f"{seed_info[0][0]},{candidate_info[0][0]},{candidate_info[0][1]},{distance}\n")