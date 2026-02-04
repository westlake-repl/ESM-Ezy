import random
import torch
from torch.utils.data import Dataset
from .fasta_dataset import FastaDataset

class TrainingDataset(Dataset):
    def __init__(self, positive_path, negative_path, dynamic_negative_sampling=False):
        super(TrainingDataset, self).__init__()
        self.positive_path = positive_path
        self.negative_path = negative_path
        self.dynamic_negative_sampling = dynamic_negative_sampling
        
        self.positive_dataset = FastaDataset(positive_path, label=1)
        self.negative_dataset = FastaDataset(negative_path, label=0)
        
    def __len__(self):
        if self.dynamic_negative_sampling:
            return len(self.positive_dataset) + len(self.positive_dataset)
        else:
            return len(self.positive_dataset) + len(self.negative_dataset)
    
    def __getitem__(self, idx):
        if idx < len(self.positive_dataset):
            return self.positive_dataset[idx]
        else:
            if self.dynamic_negative_sampling:
                return random.choice(self.negative_dataset)
            else:
                return self.negative_dataset[idx - len(self.positive_dataset)]
    
    def collate_fn(self, batch):
        sequences, labels = zip(*batch)
        ids, seqs = zip(*sequences)
        return {"ids": list(ids), "sequences": list(seqs), "labels": torch.tensor(labels)}
