import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset


class InstanceDataset(Dataset):

    def __init__(self, event_hdf5_file, event_metadata_file, noise_hdf5_file, noise_metadata_file, transform=None):
        self.event_hdf5_file = event_hdf5_file
        self.noise_hdf5_file = noise_hdf5_file
        self.noise_metadata = pd.read_csv(noise_metadata_file)
        self.event_metadata = pd.read_csv(event_metadata_file)
        self.transform = transform  

    def __len__(self):
        return self.event_metadata.shape[0] + self.noise_metadata.shape[0]
    
    def __getitem__(self, idx):
        #Return both input and target
        #TODO: Adjust target to return meaningful data

        eventsCount = self.event_metadata.shape[0]

        if(idx < eventsCount):
            #Access events
            trace_name = self.event_metadata["trace_name"][idx]
            filename = self.event_hdf5_file
            target = 1
        else:
            #Access noise
            index = idx % eventsCount
            trace_name = self.noise_metadata["trace_name"][index]
            filename = self.noise_hdf5_file
            target = 0

        with h5py.File(filename, 'r') as f:
            return  torch.tensor(f['data'][trace_name][:]), target 



