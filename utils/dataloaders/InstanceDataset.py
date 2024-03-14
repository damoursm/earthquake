import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset

class InstanceDataset(Dataset):
    def __init__(self, event_hdf5_file, event_metadata_file, noise_hdf5_file, noise_metadata_file, target_type="box_square", padding_type="sample", padding_value=280, transform=None):
        self.event_hdf5_file = event_hdf5_file
        self.noise_hdf5_file = noise_hdf5_file
        self.noise_metadata = pd.read_csv(noise_metadata_file)
        self.event_metadata = pd.read_csv(event_metadata_file, usecols=["trace_name", 'trace_start_time', 'trace_P_arrival_time', 'trace_S_arrival_time', 'trace_P_arrival_sample', 'trace_S_arrival_sample', 'trace_GPD_P_number', 'trace_GPD_S_number', 'trace_EQT_P_number', 'trace_EQT_S_number', 'trace_EQT_number_detections'])
        self.transform = transform
        self.target_type = target_type
        self.padding_type = padding_type
        self.padding_value = padding_value


    def __len__(self):
        return self.event_metadata.shape[0] + self.noise_metadata.shape[0]
    

    def __getitem__(self, idx):
        #Return both input and target
        eventsCount = self.event_metadata.shape[0]

        if(idx < eventsCount):
            #Access events
            trace_name = self.event_metadata["trace_name"][idx]
            filename = self.event_hdf5_file
            with h5py.File(filename, 'r') as f:
                input = torch.tensor(f['data'][trace_name][:])
            target = self.get_event_target(input, idx, self.event_metadata, self.target_type, self.padding_type, self.padding_value)
        else:
            #Access noise
            index = idx % eventsCount
            trace_name = self.noise_metadata["trace_name"][index]
            filename = self.noise_hdf5_file
            with h5py.File(filename, 'r') as f:
                input = torch.tensor(f['data'][trace_name][:])
            target = self.get_noise_target(input, self.target_type)

        return input, target


    def get_event_target(self, input, index, event_metadata, target_type, padding_type, padding_value):
        if target_type == "binary":
            return 1
        else:
            sample_size = input.shape[1]
            p_sample = event_metadata['trace_P_arrival_sample'][index]
            s_sample = event_metadata['trace_S_arrival_sample'][index]

            if np.isnan(p_sample):
                p_sample = 0

            if np.isnan(s_sample):
                s_sample = sample_size - 1

            s_sample = round(s_sample)

            target = torch.zeros(sample_size)
            target[p_sample:s_sample+1] = torch.ones(s_sample - p_sample + 1)

            raw_start_index, raw_end_index = self.get_raw_padding_index(p_sample, s_sample, padding_type, padding_value)

            if target_type == "box_square":
                final_start_index, start_padding, final_end_index, end_padding = self.get_box_square_padding(sample_size, raw_start_index, raw_end_index, p_sample, s_sample)
            elif target_type == "box_trapezoidal":
                final_start_index, start_padding, final_end_index, end_padding = self.get_box_trapezoidal(sample_size, raw_start_index, raw_end_index, p_sample, s_sample)
            else:
                return 1
            
            if start_padding is not None:
                target[final_start_index:p_sample] = start_padding
            
            if end_padding is not None:
                target[s_sample+1:final_end_index+1] = end_padding

            return target
            

    def get_noise_target(self, input, target_type):
        if target_type == "binary":
            return 0
        elif (target_type == "box_square") or (target_type == "box_trapezoidal"):
            return torch.zeros(input.shape[1])
        else:
            return 0 #Not optimized if/else for code clarity
        

    def get_box_square_padding(self, sample_size, raw_start_index, raw_end_index, p_sample, s_sample):
        """
        This is a box of 1s covering the sample between p and s phase in addition of a padding before and after the phases
        """
        if p_sample == 0:
            start_padding = None
            final_start_index = None
        else:
            final_start_index = max(0, raw_start_index)
            start_padding = torch.ones(p_sample - final_start_index)

        if s_sample >= sample_size - 1:
            end_padding = None
            final_end_index = None
        else:
            final_end_index = min(sample_size - 1, raw_end_index)
            end_padding = torch.ones(final_end_index - s_sample)

        return final_start_index, start_padding, final_end_index, end_padding
        

    def get_box_trapezoidal(self, sample_size, raw_start_index, raw_end_index, p_sample, s_sample):
        """
        Returns a trapezoidal target. It is an array of values between 0 and 1. Between p and s phase, it is 1s. Between start padding index it is a linespace between 0 and 1 same but reversed for end padding
        """
        if p_sample == 0:
            start_padding = None
            final_start_index = None
        else:
            padding = np.linspace(0, 1, p_sample - raw_start_index + 1) 
            final_start_index = max(0, raw_start_index)
            start_padding = torch.from_numpy(padding[final_start_index - raw_start_index:-1])

        if s_sample >= sample_size -1:
            end_padding = None
            final_end_index = None
        else:
            padding = np.flip(np.linspace(0, 1, raw_end_index - s_sample + 1))
            final_end_index = min(sample_size - 1, raw_end_index)
            end_padding = torch.from_numpy(padding[1:final_end_index - s_sample + 1].copy())

        return final_start_index, start_padding, final_end_index, end_padding

        
    def get_raw_padding_index(self, p_sample, s_sample, padding_type, padding_value):
        """
        Returns the start and end index of where detection padding should start
        """
        if padding_type == "percentage":
            padding_value = round(padding_value * (s_sample - p_sample))
        return (p_sample - padding_value), (s_sample + padding_value)

        





