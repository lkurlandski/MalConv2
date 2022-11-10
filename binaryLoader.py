import os
from collections import deque
import typing as tp
from pathlib import Path
from collections.abc import Iterable

import random
import numpy as np

from tqdm import tqdm_notebook as tqdm

import gzip

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import torch.optim as optim

from torch.utils import data

from torch.utils.data import Dataset, DataLoader, Subset


class BinaryDataset(data.Dataset):
    """
    Loader for binary files. 
    
    If you use the sort_by_file_size option, the dataset will store files from smallest to largest. This is meant to used with RandomChunkSampler to sammple batches of similarly sized files to maximize performance. 
    
    TODO: Auto un-gzip files if they have g-zip compression 
    """
    def __init__(self,
            good_dir: tp.Optional[tp.Union[str, Path, tp.Iterable]],
            bad_dir: tp.Optional[tp.Union[str, Path, tp.Iterable]],
            sort_by_size=False,
            max_len=4000000,
            shuffle=False
    ):
        
        #Tuple (file_path, label, file_size)
        self.all_files = []
        self.max_len = max_len

        if isinstance(good_dir, (str, Path)):
            for roor_dir, dirs, files in os.walk(good_dir):
                for file in files:
                    to_add = os.path.join(roor_dir,file)
                    self.all_files.append(  (to_add, 0, os.path.getsize(to_add))  )
        elif isinstance(good_dir, Iterable):
            self.all_files.extend([(f, 0, os.path.getsize(f)) for f in good_dir])

        if isinstance(bad_dir, (str, Path)):
            for roor_dir, dirs, files in os.walk(bad_dir):
                for file in files:
                    to_add = os.path.join(roor_dir,file)
                    self.all_files.append(   (to_add, 1, os.path.getsize(to_add))  )
        elif isinstance(bad_dir, Iterable):
            self.all_files.extend([(f, 1, os.path.getsize(f)) for f in bad_dir])

        if sort_by_size:
            self.all_files.sort(key=lambda filename: filename[2])

        if shuffle:
            random.shuffle(self.all_files)

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        
        to_load, y, _ = self.all_files[index]
        
        try:
            with gzip.open(to_load, 'rb') as f:
                x = f.read(self.max_len)
                #Need to use frombuffer b/c its a byte array, otherwise np.asarray will get wonked on trying to convert to ints
                #So decode as uint8 (1 byte per value), and then convert
                x = np.frombuffer(x, dtype=np.uint8).astype(np.int16)+1 #index 0 will be special padding index
        except OSError:
            #OK, you are not a gziped file. Just read in raw bytes from disk. 
            with open(to_load, 'rb') as f:
                x = f.read(self.max_len)
                #Need to use frombuffer b/c its a byte array, otherwise np.asarray will get wonked on trying to convert to ints
                #So decode as uint8 (1 byte per value), and then convert
                x = np.frombuffer(x, dtype=np.uint8).astype(np.int16)+1 #index 0 will be special padding index
            
        #x = np.pad(x, self.max_len-x.shape[0], 'constant')    
        x = torch.tensor(x)

        return x, torch.tensor([y])
    
class RandomChunkSampler(torch.utils.data.sampler.Sampler):
    """
    Samples random "chunks" of a dataset, so that items within a chunk are always loaded together. Useful to keep chunks in similar size groups to reduce runtime. 
    """
    def __init__(self, data_source, batch_size, random=True):
        """
        data_source: the souce pytorch dataset object
        batch_size: the size of the chunks to keep together. Should generally be set to the desired batch size during training to minimize runtime. 
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.random = random
        
    def __iter__(self):
        n = len(self.data_source)
        
        data = [x for x in range(n)]

        # Create blocks
        blocks = [data[i:i+self.batch_size] for i in range(0,len(data),self.batch_size)]
        # shuffle the blocks
        if self.random:
            random.shuffle(blocks)
        # concatenate the shuffled blocks
        data[:] = [b for bs in blocks for b in bs]
        
        return iter(data)
        
    def __len__(self):
        return len(self.data_source)
    
#We want to hadnel true variable length
#Data loader needs equal length. So use special function to padd all the data in a single batch to be of equal length
#to the longest item in the batch
def pad_collate_func(batch):
    """
    This should be used as the collate_fn=pad_collate_func for a pytorch DataLoader object in order to pad out files in a batch to the length of the longest item in the batch. 
    """
    vecs = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    
    x = torch.nn.utils.rnn.pad_sequence(vecs, batch_first=True)
    #stack will give us (B, 1), so index [:,0] to get to just (B)
    y = torch.stack(labels)[:,0]
    
    return x, y