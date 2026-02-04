import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pickle as pkl

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
    parser.add_argument('-m', '--model_path', type=str, default=None)
    parser.add_argument('-ckpt', '--checkpoint_path', type=str, default=None)
    parser.add_argument('-p', '--positive_path', type=str, required=True)
    parser.add_argument('-n', '--negative_path', type=str, required=True)
    parser.add_argument('-s', '--save_dir', type=str, default=None)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--shuffle', default=False, action='store_true')
    parser.add_argument('--save_every', type=int, default=1000)
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

def init_distributed():
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    print(f"World size: {world_size}, Rank: {rank}", flush=True)
    setup(rank, world_size)

def destroy_distributed():
    dist.destroy_process_group()

def setup(rank, world_size):
    dist.init_process_group(
        backend='nccl',     # 'nccl'是GPU上推荐的后端，'gloo'可以用于CPU
        init_method='env://',  # 使用环境变量来初始化进程组
        world_size=world_size,
        rank=rank
    )

def main(args=None):
    # get args
    model_path = args.model_path
    checkpoint_path = args.checkpoint_path
    esm_checkpoint = args.esm_checkpoint if model_path is None else model_path
    positive_path = args.positive_path
    negative_path = args.negative_path
    save_dir = args.save_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    shuffle = args.shuffle
    save_every = args.save_every
    
    # get rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # load model
    torch.cuda.set_device(rank)
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
    model.eval()
    
    # load dataset
    candidate_dataset = RetrievalDataset(positive_path=positive_path, negative_path=negative_path)
    sampler = DistributedSampler(candidate_dataset, num_replicas=world_size, rank=rank)
    candidate_dataloader = DataLoader(candidate_dataset, 
                                      batch_size=batch_size, 
                                      num_workers=num_workers, 
                                      shuffle=shuffle,
                                      collate_fn=candidate_dataset.collate_fn,
                                      sampler=sampler
                                    )
    print(f"Rank {rank} has {len(candidate_dataloader)} candidate sequences", flush=True)
    
    # sync all distributed processes
    dist.barrier()
    
    # get representations and labels
    # set save directory
    base_save_dir = os.path.join(save_dir, f"{os.path.basename(model_path)}|{'no' if checkpoint_path is None else 'with'}_ckpt")
    print(f"Save directory: {base_save_dir}", flush=True)
    os.makedirs(base_save_dir, exist_ok=True)
    
    # find the most largest sample idx in base_save_dir
    max_sample_idx = 0
    for file in os.listdir(base_save_dir):
        if file.startswith(f'reprs_ws{world_size}_rk{rank}') and file.endswith('.npy'):
            sample_idx = int(file.split('_')[-1].split('.')[0].split('to')[-1])
            if sample_idx > max_sample_idx:
                max_sample_idx = sample_idx
    print(f"Max sample idx in {base_save_dir} of rank {rank}: {max_sample_idx}", flush=True)
    
    total_labels = []
    total_reprs = []
    total_names = []
    save_last = 0
    count = 0
    for idx, batch in enumerate(tqdm(candidate_dataloader)):
        sample_idx = (idx+1) * batch_size
        if sample_idx <= max_sample_idx:
            save_last = sample_idx
            continue
        sequences = batch["sequences"]
        labels = batch["labels"]
        count += len(sequences)
        # collect labels in a list
        total_labels.extend(labels)
        # collect names in a list
        names = batch["ids"]
        total_names.extend(names)
        with torch.no_grad():
            representations = model.get_representations(batch)
            total_reprs.append(representations.cpu().numpy())
            
        # save representations and labels every 1000 batches
        if (idx+1) % save_every == 0:
            _total_reprs = np.concatenate(total_reprs, axis=0)
            _total_labels = np.array(total_labels)
            dist.barrier()
            np.save(os.path.join(base_save_dir, f'reprs_ws{world_size}_rk{rank}_sample{save_last}to{sample_idx}.npy'), _total_reprs)
            np.save(os.path.join(base_save_dir, f'labels_ws{world_size}_rk{rank}_sample{save_last}to{sample_idx}.npy'), _total_labels)
            pkl.dump(total_names, open(os.path.join(base_save_dir, f'names_ws{world_size}_rk{rank}_sample{save_last}to{sample_idx}.pkl'), 'wb'))
            save_last = sample_idx
            total_reprs = []
            total_labels = []
            total_names = []

    sample_idx = (idx+1) * batch_size
    total_reprs = np.concatenate(total_reprs, axis=0)
    total_labels = np.array(total_labels)
    dist.barrier()
    np.save(os.path.join(base_save_dir, f'reprs_ws{world_size}_rk{rank}_sample{save_last}to{sample_idx}.npy'), total_reprs)
    np.save(os.path.join(base_save_dir, f'labels_ws{world_size}_rk{rank}_sample{save_last}to{sample_idx}.npy'), total_labels)
    pkl.dump(total_names, open(os.path.join(base_save_dir, f'names_ws{world_size}_rk{rank}_sample{save_last}to{sample_idx}.pkl'), 'wb'))
    
    print(f"Total datas in rank {rank}: {count}", flush=True)
    
    dist.barrier()
    

if __name__ == '__main__':
    args = parse_args()
    init_distributed()
    print(args)
    main(args)
    destroy_distributed()
