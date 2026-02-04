import torch
from torch.utils.data import Dataset
from utils.fasta import read_fasta

class FastaDataset(Dataset):
    def __init__(self, fasta_file, label=None):
        super(FastaDataset, self).__init__()
        self.fasta_file = fasta_file
        self.label = label
        
        self.fasta_list = self.read_fasta_file(fasta_file)
    
    def read_fasta_file(self, fasta_file):
        return read_fasta(fasta_file)

    def __getitem__(self, index):
        if self.label is None:
            return self.fasta_list[index]
        else:
            return self.fasta_list[index], self.label
    
    def __len__(self):
        return len(self.fasta_list)
    
    def collate_fn(self, batch):
        ids, sequences = zip(*batch)
        return {"ids": list(ids), "sequences": list(sequences)}
