import h5py
import numpy as np
import utils.plot
import pandas as pd


# Load data, I would only keep this function for testing purposes
def load_dataset(filename):
    print("load_dataset running")
    with h5py.File(filename, 'r') as f:
        if 'data' in f:
            data = f['data']
            if isinstance(data, h5py.Dataset):
                print("Type du dataset 'data':", type(data))
                print("Forme du dataset 'data':", data.shape)
                data_matrix = data[:]
                print("Matrice de données:", data_matrix)
                return data_matrix
            elif isinstance(data, h5py.Group):
                print("Type du dataset 'data':", type(data))
                data_list = []
                trace_names = []
                for key in data.keys():
                    dataset = data[key]
                    if isinstance(dataset, h5py.Dataset):
                        # Vérifier la forme des données extraites
                        if dataset.ndim == 2:
                            # print(dataset.shape[0])
                            data_array = np.zeros((1, dataset.shape[0], dataset.shape[1]),
                                                  dtype=np.float32)  # Créer une matrice vide de trois dimensions
                            for i in range(3):  # Boucle à travers les canaux
                                data_array[:, i, :] = dataset[i, :]  # Récupérer les valeurs des trois canaux existants
                            data_list.append(data_array)
                            # trace_names.append(key)
                            # Concaténer les noms de clés avec un tableau de zéros
                            trace_names = np.concatenate([np.array([key]).reshape(-1, 1) for key in data.keys()])
                            # Remodeler en un vecteur colonne
                            trace_names = trace_names.reshape(-1, 1)
                            trace_names = trace_names[:, 0]
                            # trace_names['trace_names']=trace_names

                        else:
                            print(f"Les données pour la clé '{key}' ne sont pas sous forme de tableau 2D.")
                print("Finished keys")
                if data_list:
                    stacked_data_matrix = np.stack(data_list, axis=0)
                    # Redimensionnement des données
                    stacked_data_matrix = np.squeeze(stacked_data_matrix, axis=1)

                    # Vérification du nouveau shape
                    # print(data_resized.shape)
                    return stacked_data_matrix, trace_names


# Augment noise (times 8) one file at a time (on the EQTransformer GitHub, they seem to augment data in batches, but I
# ran into more problems doing so).
def augmentation_noise(noise, noise_names):
    """
    noise : initial noise data
    noise_names : initial noise names
    returns augmented noise data, augmented noise data names
    """
    all_noise = []
    all_noise_names = []

    for j in range(len(noise)):  # For every noise
        noise_elem = noise[j]
        noise_elem_name = noise_names[j]
        for i in range(8):  # Multiply number of noise by 8
            noise_cp = np.copy(noise_elem)
            noise_cp = _drop_channel_noise(noise_cp, i % 2 == 0)
            noise_cp = _add_gaps(noise_cp, i < 4)
            noise_cp = _add_noise(noise_cp, i % 4 < 2)
            # These second parameters make it so the configuration for an augment is always unique
            noise_cp = _normalize(noise_cp)  # Always normalize
            all_noise.append(noise_cp)
            all_noise_names.append(noise_elem_name + str(i))

            # For visualisation
            # utils.plot.plot_trace_prediction(all_noise_names[i + 8 * j], all_noise[i + 8 * j], [0], [0], [0], [0],
            # [0], "../output/figures")

    all_noise = np.array(all_noise)
    all_noise_names = np.array(all_noise_names)

    return all_noise, all_noise_names


# Augment noise by batch (had some problems, keeping it just in case)
"""def augmentation_noise_batch(noise, noise_names, drop_rate=0.5, gap_rate=0.5, noise_rate=0.5, snr=20, add_name='mod'):
    
    #noise : initial noise data
    #noise_names : initial noise names
    #returns augmented noise data + augmented noise data names
    
    noise_cp = np.copy(noise)
    noise_cp = _drop_channel_noise(noise_cp, drop_rate)
    noise_cp = _add_gaps(noise_cp, gap_rate)
    # noise_cp = _add_noise(noise_cp, snr, noise_rate)
    noise_cp = _normalize(noise_cp)
    noise_names = np.char.add(noise_names, add_name)
    return noise_cp, noise_names


# General noise augmentation function
def noise_augment_general(noise, noise_names, augment=10):
    all_noise = np.empty(np.shape(noise))
    all_names = np.empty(np.shape(noise_names))
    for i in range(augment):
        noise_augmented_matrix, noise_augmented_names = augmentation_noise_batch(noise, noise_names, add_name=str(i))
        if i == 0:
            all_noise = noise_augmented_matrix
            all_names = noise_augmented_names
        else:
            all_noise = np.concatenate((all_noise, noise_augmented_matrix))
            all_names = np.concatenate((all_names, noise_augmented_names))
    visualize_batch(all_noise, all_names)
    return all_noise, all_names"""


# Visualize a batch of noise with utils.plot_trace_prediction()
def visualize_batch(noise, noise_names):
    for i in range(len(noise)):
        utils.plot.plot_trace_prediction(noise_names[i][0], noise[i], [0], [0], [0], [0], [0], "../output/figures")
    return


def _normalize(data, mode='max'):
    'Normalize waveforms in each batch'

    data -= np.mean(data, axis=0, keepdims=True)
    if mode == 'max':
        max_data = np.max(data, axis=0, keepdims=True)
        assert (max_data.shape[-1] == data.shape[-1])
        max_data[max_data == 0] = 1
        data /= max_data

    elif mode == 'std':
        std_data = np.std(data, axis=0, keepdims=True)
        assert (std_data.shape[-1] == data.shape[-1])
        std_data[std_data == 0] = 1
        data /= std_data
    return data


def _scale_amplitude(data, rate):
    'Scale amplitude or waveforms'

    tmp = np.random.uniform(0, 1)
    if tmp < rate:
        data *= np.random.uniform(1, 3)
    elif tmp < 2 * rate:
        data /= np.random.uniform(1, 3)
    return data


def _drop_channel(data, snr, rate):
    'Randomly replace values of one or two components to zeros in earthquake data'

    data = np.copy(data)
    if np.random.uniform(0, 1) < rate and all(snr >= 10.0):
        c1 = np.random.choice([0, 1])
        c2 = np.random.choice([0, 1])
        c3 = np.random.choice([0, 1])
        if c1 + c2 + c3 > 0:
            data[..., np.array([c1, c2, c3]) == 0] = 0
    return data


def _drop_channel_noise(data, rate):
    'Randomly replace values of one or two components to zeros in noise data'

    data = np.copy(data)
    if np.random.uniform(0, 1) < rate:
        c1 = np.random.choice([0, 1])
        c2 = np.random.choice([0, 1])
        c3 = np.random.choice([0, 1])
        if c1 + c2 + c3 > 0:
            data[np.array([c1, c2, c3]) == 0] = 0
    return data


def _add_gaps(data, rate):
    'Randomly add gaps (zeros) of different sizes into waveforms'

    data = np.copy(data)
    # À changer les range selon nos données, Instance ---> 12 000
    gap_start = np.random.randint(0, 8000)  # Was 0, 4000 for STEAD
    gap_end = np.random.randint(gap_start, 10500)  # Was 5500 for STEAD
    if np.random.uniform(0, 1) < rate:
        data[gap_start:gap_end, :] = 0
    return data


def _add_noise(data, rate):
    'Randomly add Gaussian noise with a random SNR into a waveform'

    data_noisy = np.empty((data.shape))
    if np.random.uniform(0, 1) < rate:
        data_noisy = np.empty((data.shape))
        data_noisy[0, :] = data[0, :] + np.random.normal(0, np.random.uniform(0.01, 0.05) * max(data[0, :]),
                                                         data.shape[1])
        data_noisy[1, :] = data[1, :] + np.random.normal(0, np.random.uniform(0.01, 0.15) * max(data[1, :]),
                                                         data.shape[1])
        data_noisy[2, :] = data[2, :] + np.random.normal(0, np.random.uniform(0.01, 0.15) * max(data[2, :]),
                                                         data.shape[1])
    else:
        data_noisy = data
    return data_noisy


def _adjust_amplitude_for_multichannels(data):
    'Adjust the amplitude of multichaneel data'

    tmp = np.max(np.abs(data), axis=0, keepdims=True)
    assert (tmp.shape[-1] == data.shape[-1])
    if np.count_nonzero(tmp) > 0:
        data *= data.shape[-1] / np.count_nonzero(tmp)
    return data


def _add_event(data, addp, adds, coda_end, snr, rate):
    'Add a scaled version of the event into the empty part of the trace'

    added = np.copy(data)
    additions = None
    spt_secondEV = None
    sst_secondEV = None
    if addp and adds:
        s_p = adds - addp
        if np.random.uniform(0, 1) < rate and all(snr >= 10.0) and (data.shape[0] - s_p - 21 - coda_end) > 20:
            secondEV_strt = np.random.randint(coda_end, data.shape[0] - s_p - 21)
            scaleAM = 1 / np.random.randint(1, 10)
            space = data.shape[0] - secondEV_strt
            added[secondEV_strt:secondEV_strt + space, 0] += data[addp:addp + space, 0] * scaleAM
            added[secondEV_strt:secondEV_strt + space, 1] += data[addp:addp + space, 1] * scaleAM
            added[secondEV_strt:secondEV_strt + space, 2] += data[addp:addp + space, 2] * scaleAM
            spt_secondEV = secondEV_strt
            if spt_secondEV + s_p + 21 <= data.shape[0]:
                sst_secondEV = spt_secondEV + s_p
            if spt_secondEV and sst_secondEV:
                additions = [spt_secondEV, sst_secondEV]
                data = added

    return data, additions


def _shift_event(data, addp, adds, coda_end, snr, rate):
    'Randomly rotate the array to shift the event location'

    org_len = len(data)
    data2 = np.copy(data)
    addp2 = adds2 = coda_end2 = None
    if np.random.uniform(0, 1) < rate:
        nrotate = int(np.random.uniform(1, int(org_len - coda_end)))
        data2[:, 0] = list(data[:, 0])[-nrotate:] + list(data[:, 0])[:-nrotate]
        data2[:, 1] = list(data[:, 1])[-nrotate:] + list(data[:, 1])[:-nrotate]
        data2[:, 2] = list(data[:, 2])[-nrotate:] + list(data[:, 2])[:-nrotate]

        if addp + nrotate >= 0 and addp + nrotate < org_len:
            addp2 = addp + nrotate
        else:
            addp2 = None
        if adds + nrotate >= 0 and adds + nrotate < org_len:
            adds2 = adds + nrotate
        else:
            adds2 = None
        if coda_end + nrotate < org_len:
            coda_end2 = coda_end + nrotate
        else:
            coda_end2 = org_len
        if addp2 and adds2:
            data = data2
            addp = addp2
            adds = adds2
            coda_end = coda_end2
    return data, addp, adds, coda_end


# Example of use for noise augmentation
# noise_matrix, trace_names_noise = load_dataset('../Instance_noise_1k.hdf5')  # Get data
# noise_augmented_matrix, noise_augmented_names = augmentation_noise(noise_matrix[:3], trace_names_noise[:3])  # Augment



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

