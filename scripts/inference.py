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
    parser.add_argument('--inference_data', type=str, default=None)
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
    inference_data = args.inference_data
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

    # data
    inference_dataloader = None
    ids_only = None
    if args.embedding_source == "precomputed" and inference_data is None:
        if args.precomputed_embeddings_dir is None or not args.precomputed_embeddings_dir.endswith(".npz"):
            raise ValueError("For precomputed embeddings without inference_data, provide a .npz container file.")
        npz = np.load(args.precomputed_embeddings_dir, allow_pickle=False)
        ids_only = list(npz.files)
    else:
        print("Reading candidate data...")
        inference_dataset = FastaDataset(inference_data)
        inference_dataloader = DataLoader(
            inference_dataset,
            batch_size=64,
            shuffle=True,
            collate_fn=inference_dataset.collate_fn,
            drop_last=False,
            pin_memory=True,
        )
    
    inference_list = []
    inference_ids = []
    with torch.no_grad():
        if ids_only is not None:
            batch_size = 64
            for start in tqdm(range(0, len(ids_only), batch_size)):
                batch = {"ids": ids_only[start:start + batch_size], "sequences": None}
                last_result = model(batch)
                mask = last_result[:, 1] > last_result[:, 0]
                inference_ids.extend([i for m, i in zip(mask, batch["ids"]) if m])
        else:
            for batch in tqdm(inference_dataloader, total=len(inference_dataloader)):
                last_result = model(batch)
                mask = last_result[:, 1] > last_result[:, 0]
                inference_list.extend(
                    [(i, s) for m, i, s in zip(mask, batch["ids"], batch["sequences"]) if m]
                )

    if inference_list:
        with open(os.path.join(output_path, "candidate.fa"), "w") as f:
            for c in inference_list:
                f.write(f">{c[0]}\n{c[1]}\n")
    else:
        with open(os.path.join(output_path, "candidate_ids.txt"), "w") as f:
            for i in inference_ids:
                f.write(f"{i}\n")
