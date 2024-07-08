import torch
from torch.utils.data import Dataset
from .fasta_dataset import FastaDataset

class RetrievalDataset(Dataset):
    def __init__(self, positive_path, negative_path):
        super(RetrievalDataset, self).__init__()
        self.positive_path = positive_path
        self.negative_path = negative_path
        
        self.positive_dataset = FastaDataset(positive_path, label=1)
        self.negative_dataset = FastaDataset(negative_path, label=0)
        
    def __len__(self):
        return len(self.positive_dataset) + len(self.negative_dataset)
    
    def __getitem__(self, idx):
        if idx < len(self.positive_dataset):
            return self.positive_dataset[idx]
        else:
            return self.negative_dataset[idx - len(self.positive_dataset)]
    
    def collate_fn(self, batch):
        sequences, labels = zip(*batch)
        return sequences, torch.tensor(labels)