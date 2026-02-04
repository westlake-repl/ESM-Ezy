import torch
from torch.utils.data import Dataset
from .fasta_dataset import FastaDataset

class RetrievalDataset(Dataset):
    def __init__(self, candidate_path=None, seed_path=None, positive_path=None, negative_path=None):
        super(RetrievalDataset, self).__init__()
        if candidate_path is None and positive_path is not None:
            candidate_path = positive_path
        if seed_path is None and negative_path is not None:
            seed_path = negative_path
        if candidate_path is None or seed_path is None:
            raise ValueError("candidate_path/seed_path (or positive_path/negative_path) must be provided.")

        self.candidate_path = candidate_path
        self.seed_path = seed_path

        self.candidate_dataset = FastaDataset(candidate_path, label=1)
        self.seed_dataset = FastaDataset(seed_path, label=0)

    def __len__(self):
        return len(self.candidate_dataset) + len(self.seed_dataset)

    def __getitem__(self, idx):
        if idx < len(self.candidate_dataset):
            return self.candidate_dataset[idx]
        return self.seed_dataset[idx - len(self.candidate_dataset)]

    def collate_fn(self, batch):
        sequences, labels = zip(*batch)
        ids, seqs = zip(*sequences)
        return {"ids": list(ids), "sequences": list(seqs), "labels": torch.tensor(labels)}
