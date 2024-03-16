import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset
import math

class InstanceDataset(Dataset):
    """
    Raw Instance dataset.
    Load events and noise from file and returns the 3 C waveforms, wether there is a earthquake or not, p phase and s phase
    """
    def __init__(self, event_hdf5_file, event_metadata_file, noise_hdf5_file, noise_metadata_file, target_type="box_square", padding_type="sample", padding_value=280, target_phase=True, phase_padding=20, remove_empty_phase=True, transform=None, split_index=0, split_percentage=[0.8, 0.15, 0.5]):
        self.event_hdf5_file = event_hdf5_file
        self.noise_hdf5_file = noise_hdf5_file
        self.noise_metadata = pd.read_csv(noise_metadata_file)
        self.event_metadata = pd.read_csv(event_metadata_file, usecols=["trace_name", 'trace_start_time', 'trace_P_arrival_time', 'trace_S_arrival_time', 'trace_P_arrival_sample', 'trace_S_arrival_sample', 'trace_GPD_P_number', 'trace_GPD_S_number', 'trace_EQT_P_number', 'trace_EQT_S_number', 'trace_EQT_number_detections'])
        if remove_empty_phase:
            self.event_metadata = self.event_metadata[self.event_metadata["trace_P_arrival_sample"].notna() & self.event_metadata["trace_S_arrival_sample"].notna()]
        # self.event_metadata = self.event_metadata.sample(1, random_state=1)
        # self.noise_metadata = self.noise_metadata.sample(1, random_state=1)
        self.transform = transform
        self.target_type = target_type
        self.padding_type = padding_type
        self.padding_value = padding_value
        self.target_phase = target_phase
        self.phase_padding = phase_padding

        self.event_start_index = 0
        self.noise_start_index = 0
        for i in range(split_index):
            self.event_start_index = self.event_start_index + math.floor(self.event_metadata.shape[0] * split_percentage[i])
            self.noise_start_index = self.noise_start_index + math.floor(self.noise_metadata.shape[0] * split_percentage[i])
        
        self.events_size=math.floor(self.event_metadata.shape[0] * split_percentage[split_index])
        self.noise_size=math.floor(self.noise_metadata.shape[0] * split_percentage[split_index])


    def __len__(self):
        return self.events_size + self.noise_size
    

    def __getitem__(self, idx):
        """
        Returns 4 items:
        1. input data: 3 channel waveforms x number of counts
        2. detection target: 0/1 for binary type, torch array of 0s with 1s between S and P phase + padding of other square and trapeze types
        3. p phase target: -1 for binary type (not supported), torch array of 0s with a 1 on the exact P phase sample and padding values between 0-1 
        3. s phase target: -1 for binary type (not supported), torch array of 0s with a 1 on the exact S phase sample and padding values between 0-1 
        """
        if(idx < self.events_size):
            index = idx + self.event_start_index
            print(index)
            #Access events
            trace_name = self.event_metadata["trace_name"].iloc[index]
            print(trace_name)
            filename = self.event_hdf5_file
            with h5py.File(filename, 'r') as f:
                input = torch.tensor(f['data'][trace_name][:])
            target, p_target, s_target = self.get_event_target(input, index, self.event_metadata, self.target_type, self.padding_type, self.padding_value, self.target_phase, self.phase_padding)
        else:
            #Access noise
            index = (idx % self.events_size) + self.noise_start_index
            print(index)
            trace_name = self.noise_metadata["trace_name"].iloc[index]
            print(trace_name)
            filename = self.noise_hdf5_file
            with h5py.File(filename, 'r') as f:
                input = torch.tensor(f['data'][trace_name][:])
            target, p_target, s_target = self.get_noise_target(input, self.target_type, self.target_phase)

        return input, target, p_target, s_target


    def get_event_target(self, input, index, event_metadata, target_type, padding_type, padding_value, target_phase, phase_padding):
        s_target = -1
        p_target = -1
        
        if target_type == "binary":
            target = 1
        else:
            sample_size = input.shape[1]
            p_sample = event_metadata['trace_P_arrival_sample'].iloc[index]
            s_sample = event_metadata['trace_S_arrival_sample'].iloc[index]

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
                start_padding = None
                end_padding = None
                target = 1
            
            if start_padding is not None:
                target[final_start_index:p_sample] = start_padding
            
            if end_padding is not None:
                target[s_sample+1:final_end_index+1] = end_padding

            if target_phase and ((target_type == "box_square") or (target_type == "box_trapezoidal")):
                p_target = self.get_phase_target(event_metadata['trace_P_arrival_sample'].iloc[index], sample_size, phase_padding) 
                s_target = self.get_phase_target(event_metadata['trace_S_arrival_sample'].iloc[index], sample_size, phase_padding)
            
        return target, p_target, s_target
    

    def get_phase_target(self, phase_sample, sample_size, phase_padding):
        target = torch.zeros(sample_size)
        if not np.isnan(phase_sample):
            phase_sample = round(phase_sample)
            start_index = phase_sample - phase_padding
            padding = np.linspace(0, 1, phase_padding + 1)
            final_start_index = max(0, start_index)
            target[final_start_index:phase_sample+1] = torch.from_numpy(padding[final_start_index - start_index:])

            end_index = phase_sample + phase_padding
            padding = np.flip(np.linspace(0, 1, phase_padding + 1))
            final_end_index = min(sample_size - 1, end_index)
            target[phase_sample+1:final_end_index+1] = torch.from_numpy(padding[1:final_end_index - phase_sample + 1].copy())
        return target


    def get_noise_target(self, input, target_type, target_phase):
        target = 0
        phase_target = -1

        if (target_type == "box_square") or (target_type == "box_trapezoidal"):
            target = torch.zeros(input.shape[1])
            if target_phase:
                phase_target = torch.zeros(input.shape[1])
    
        return target, phase_target, phase_target



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

        





