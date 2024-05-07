import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
def seq_to_data(seq):
    N = len(seq)
    dum = np.ndarray(N)
    for idx,i in enumerate(seq):
        num = ord(i) - 64
        dum[idx] = num 
    a = torch.IntTensor(dum)
    return a
def label_to_data(label):
    N = len(label)
    dum = np.ndarray(N)
    for idx,i in enumerate(label):
        if  i == '0':
            dum[idx] = 0
        if i == "1":
            dum[idx] = 1
    a = torch.IntTensor(dum)
    return a
class ProteinDataset(data.Dataset):
    def __init__(self, path):
        sequences = []
        labels = []
        with open(path,'r') as file:
            for idx,line in enumerate(file):
                #sequence
                if idx%3 == 1 :
                    sequences.append(seq_to_data(line.strip()))
                #label    
                if idx%3 == 2 :
                    labels.append(label_to_data(line.strip()))
        self.sequences = sequences
        self.labels = labels
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        return torch.IntTensor(sequence), torch.tensor(label)