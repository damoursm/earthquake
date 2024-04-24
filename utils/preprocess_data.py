import pandas as pd

def shuffle_events(event_metadata_file, noise_metadata_file, random_state, output_event_metadata_file, output_noise_metadata_file, balance=False, remove_empty_phase=False):
    
    """ 

    Split the list of input data into training, validation, and test set.

    Parameters
    ----------
    event_metadata_file: str
        path to event metadata file 
    
    noise_metadata_file: str
        path to event metadata file 

    random_state: int
        random state for reproducibility

    output_event_metadata_file: bool
       Path to the output event metadata file

    output_noise_metadata_file: str
       Path to the output noise metadata file

    balance: bool
       Should have equal noise and events 

    remove_empty_phase: bool
       Remove events that do not have both phases
              
    Returns
    -------   
    (output_event_metadata_file, output_noise_metadata_file)
    """       

    noise_metadata = pd.read_csv(noise_metadata_file)
    event_metadata = pd.read_csv(event_metadata_file, usecols=["trace_name", 'trace_start_time', 'trace_P_arrival_time', 'trace_S_arrival_time', 'trace_P_arrival_sample', 'trace_S_arrival_sample', 'trace_GPD_P_number', 'trace_GPD_S_number', 'trace_EQT_P_number', 'trace_EQT_S_number', 'trace_EQT_number_detections'])

    if remove_empty_phase:
        event_metadata = event_metadata[event_metadata["trace_P_arrival_sample"].notna() & event_metadata["trace_S_arrival_sample"].notna()]
    
    event_metadata = event_metadata.sample(frac=1, random_state=random_state) 
    noise_metadata =noise_metadata.sample(frac=1, random_state=random_state) 

    if balance:
        min_count = min(event_metadata.shape[0], noise_metadata.shape[0])
        event_metadata = event_metadata.head(min_count)
        noise_metadata = noise_metadata.head(min_count)

    event_metadata.to_csv(output_event_metadata_file, index=False)
    noise_metadata.to_csv(output_noise_metadata_file, index=False)

    return output_event_metadata_file, output_noise_metadata_file 

