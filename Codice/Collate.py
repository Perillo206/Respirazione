import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
def collate3_fn(batch, max_len):
    sequences = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    min_vals = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    max_vals = torch.tensor([item[3] for item in batch], dtype=torch.float32)
    file_names = [item[4] for item in batch]

    # Pad sequences tutti a max_len (assumendo che sequences siano tensori (seq_len, features))
    padded_sequences = []
    for seq in sequences:
        seq_len = seq.shape[0]
        if seq_len < max_len:
            pad_size = max_len - seq_len
            # padding con zeri alla fine, shape (pad_size, features)
            padding = torch.zeros(pad_size, seq.shape[1])
            padded_seq = torch.cat((seq, padding), dim=0)
        else:
            padded_seq = seq[:max_len]  # taglia se troppo lunga
        padded_sequences.append(padded_seq)

    inputs = torch.stack(padded_sequences)  # shape (batch, max_len, features)
    return inputs, labels, min_vals, max_vals, file_names
