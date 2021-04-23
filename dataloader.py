# preliminary imports

import pandas as pd
import torch
import logging
import time

from torch.utils.data import DataLoader, Dataset
from easydict import EasyDict as edict


# create dataset
class LingFeatDataset(Dataset):
    def __init__(self, df):
        self.df = df
        
    def num_class(self):
        # Return number of classes
        return len(self.df['Grade'].unique())
        
    def __len__(self):
        # Number of rows
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        # Retreive 'Text', 'Grade', 'Index' from dataframe
        source = self.df['Text'].values[idx]
        target = self.df['Grade'].values[idx]
        item_idx = self.df.index[idx]
        return {
            'source': source,
            'target': target,
            'item_idx': item_idx
        }

# collate function
class LingFeatBatchGenerator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        sources = [item['source'] for item in batch]
        targets = [item['target'] for item in batch]
        item_idxs = [item['item_idx'] for item in batch]    # map item during inference stage

        inputs = self.tokenizer(
            sources,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        labels = torch.LongTensor(targets)
        
        return inputs, labels, item_idxs
