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
import os
try:
    import faiss
    _FAISS_AVAILABLE = True
except Exception:
    faiss = None
    _FAISS_AVAILABLE = False
try:
    import mkl
    mkl.get_max_threads()
except Exception:
    pass

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--candidate_data', type=str)
    parser.add_argument('--seed_data', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--embedding_source', type=str, default="fair_esm2", choices=["fair_esm2", "precomputed"])
    parser.add_argument('--esm_checkpoint', type=str, default="esm2_t36_3B_UR50D")
    parser.add_argument('--device', type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument('--precision', type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument('--precomputed_embeddings_dir', type=str, default=None)
    parser.add_argument('--precomputed_format', type=str, default="auto", choices=["auto", "npy", "npz", "pt"])
    parser.add_argument('--precomputed_granularity', type=str, default="per_sequence",
                        choices=["per_sequence", "per_residue"])
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = parse_args()
    model_path = args.model_path
    checkpoint_path = args.checkpoint_path
    esm_checkpoint = args.esm_checkpoint if model_path is None else model_path
    candidate_data = args.candidate_data
    seed_data = args.seed_data
    output_path = args.output_path
    
    # load model
    print("Loading model...")
    model = LaccaseModel.from_pretrained(
        pretrained_model_path=esm_checkpoint,
        state_dict_path=checkpoint_path,
        embedding_source=args.embedding_source,
        esm_checkpoint=esm_checkpoint,
        device=args.device,
        precision=args.precision,
        precomputed_embeddings_dir=args.precomputed_embeddings_dir,
        precomputed_format=args.precomputed_format,
        precomputed_granularity=args.precomputed_granularity,
    )
    model = model.to(model.device)
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
        for j, batch in tqdm(enumerate(candidate_dataloader), total=len(candidate_dataloader)):
            out_result, last_repr = model(batch, return_repr=True)
            for i, s, r in zip(batch["ids"], batch["sequences"], last_repr.cpu().numpy()):
                candidate_info_list.append((i, s, r))
    candidate_repr = np.stack([r for _, _, r in candidate_info_list], axis=0)
    print(candidate_repr.shape)


    use_faiss = os.environ.get("ESM_EZY_USE_FAISS", "1") != "0" and _FAISS_AVAILABLE
    if use_faiss:
        index = faiss.IndexFlatL2(model.d_model)
        index.add(candidate_repr.astype("float32"))

    result_list = []
    with torch.no_grad():
        for j, batch in tqdm(enumerate(seed_dataloader), total=len(seed_dataloader)):
            out_result, last_repr = model(batch, return_repr=True)
            seed_repr = last_repr.cpu().numpy().astype("float32")
            if use_faiss:
                D, I = index.search(seed_repr, k=10)
                pairs = zip(I[0], D[0])
            else:
                diff = candidate_repr.astype("float32") - seed_repr[0]
                dists = (diff * diff).sum(axis=1)
                topk_idx = dists.argsort()[:10]
                pairs = [(i, float(dists[i])) for i in topk_idx]
            for i, distance in pairs:
                res_tuple = ((batch["ids"][0], batch["sequences"][0]), candidate_info_list[i], distance)
                if res_tuple not in result_list:
                    result_list.append(res_tuple)

    with open(os.path.join(output_path, "results.csv"), "w") as f:
        f.write("seed_id,candidate_id,candidate_sequence,distance\n")
        for res in result_list:
            seed_info, candidate_info, distance = res
            f.write(f"{seed_info[0]},{candidate_info[0]},{candidate_info[1]},{distance}\n")
