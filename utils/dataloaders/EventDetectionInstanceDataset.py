from utils.dataloaders.InstanceDataset import InstanceDataset

class EventDetectionInstanceDataset(InstanceDataset):
    """
    Loads instance data from file and returns the 3C waveforms and wether the signal is a earthquake or not
    """

    def __init__(self, event_hdf5_file, event_metadata_file, noise_hdf5_file, noise_metadata_file, target_type="box_square", padding_type="sample", start_padding_value=280, end_padding_value=280, transform=None, split_index=0, split_percentage=[0.8, 0.15, 0.5]):
        super().__init__(event_hdf5_file, event_metadata_file, noise_hdf5_file, noise_metadata_file, target_type, padding_type, start_padding_value, end_padding_value, False, 0, False, transform, split_index, split_percentage)

    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, idx):
        data, detection, _, _, _ = super().__getitem__(idx)
        return data, detection