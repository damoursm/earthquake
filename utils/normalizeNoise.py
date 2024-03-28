import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler

"""
Ce code sert principalement a normalizer les donnees et ajouter du bruit gaussien en utilisant principes
de SNR. Il est /galement possible de couper les signaux pour prendre justes certain intervalles ou merger les noises et earthaquake dans le 
meme hdf5 en runnant la fonction avec HDF_TARGET = Noise et HDF_TARGET = Earthquake. La fonction creer un nouveau fichier HDF5 avec ce qui a ete specifier.

****   SI LE FICHIER EXISTE DEJA LA FONCTION APPEND A LA PLACE DONC NE PAS RUN SUR LE DATASET COMPLET POUR EVITER DE MODIFIER LA DATABASE  ****
NomFichierOutput = "NE_PAS_CHOISIR_LA_DATA_BASE.hdf5"     
"""


# Dans le hdf5 les signaux sont dans une subsection "data" et les code de chaque signal suit la nomenclature du
# fichier csv (source_id,station_network_code,station_code,station_location_code,station_channels).
# Les signaux sont de 120 seconde et 100hz.
# # composant ce qui nous donne 3 * 12000 points par earthquake 
# si on veut un sample des 20 premiere seconde il faut donc prendre par exemple des 2000 premier points (100 points par secondes)

def hdf5_creation(url_event_hdf5, debut_point, max_point, chunk_size, sample_size, target_snr):
    
    #stats_df = pd.DataFrame(columns=["mean_E", "median_E", "max_E", "min_E",
                                     #"mean_N", "median_N", "max_N", "min_N",
                                     #"mean_Z", "median_Z", "max_Z", "min_Z"])
    
    if hdf_target == nom_fichier_output:
        return 
    
    with h5py.File(url_event_hdf5, 'r') as hdf5:
        new_noise_data = {}
        for i, nom_dataset in enumerate(hdf5['data']):
            if i >= sample_size:
                break
            data_set = hdf5['data'][nom_dataset]
            rows, cols = data_set.shape
            new_noise_data = {}
            if url_event_hdf5 == earthquake:
                noise_data = process_data(data_set, debut_point, max_point, chunk_size, target_snr)
                new_noise_data[f"{nom_dataset}_earthquake"] = noise_data
            elif url_event_hdf5 == noise:
                noise_data = process_data(data_set, debut_point, max_point, chunk_size, target_snr, nombre_noise)
                new_noise_data[f"{nom_dataset}"] = noise_data
            append_noise_data_to_hdf5(nom_fichier_output, new_noise_data)
            
            #stats_df = calculate_and_append_stats(stats_df, noise_data, i)
            #stats_df.to_csv('test.csv', index=False)
            
            
def process_data(data_set, debut_point, max_point, chunk_size, target_snr, nombre_noise=1):
    rows, cols = data_set.shape
    noise_data = np.empty((rows, max_point - debut_point))
    for j in range(nombre_noise):
        for start in range(0, rows, chunk_size):
            end = min(start + chunk_size, rows)
            chunk = data_set[start:end, debut_point:max_point]
            chunk = normalize_signal(chunk)
            adjusted_chunk = adjust_noise_level(chunk, target_snr)
            noise_data[start:end, :] = adjusted_chunk
    return noise_data
"""
Le code pour calculer les stats est dans un fichier stats_csv.py 

def calculate_and_append_stats(stats_df, noise_data, index):
    mean_E = np.mean(noise_data[0], axis=0)
    median_E = np.median(noise_data[0], axis=0)
    max_E = np.max(noise_data[0], axis=0)
    min_E = np.min(noise_data[0], axis=0)
    mean_N = np.mean(noise_data[1], axis=0)
    median_N = np.median(noise_data[1], axis=0)
    max_N = np.max(noise_data[1], axis=0)
    min_N = np.min(noise_data[1], axis=0)
    mean_Z = np.mean(noise_data[2], axis=0)
    median_Z = np.median(noise_data[2], axis=0)
    max_Z = np.max(noise_data[2], axis=0)
    min_Z = np.min(noise_data[2], axis=0)

    new_stats_df = pd.DataFrame({
        "mean_E": [mean_E], "median_E": [median_E], "max_E": [max_E], "min_E": [min_E],
        "mean_N": [mean_N], "median_N": [median_N], "max_N": [max_N], "min_N": [min_N],
        "mean_Z": [mean_Z], "median_Z": [median_Z], "max_Z": [max_Z], "min_Z": [min_Z]
        }, index=[index])
    return pd.concat([stats_df, new_stats_df])

"""
def calculate_snr(matrix):
    snr_values = []
    for row in matrix:
        signal_power = np.sum(row ** 2)
        noise_power = np.sum((row - np.mean(row)) ** 2)
        snr = 10 * np.log10(signal_power / noise_power)
        snr_values.append(snr)
    avg_snr = np.mean(snr_values)
    return avg_snr

# ajuster le noise
# Il y a des matrica Nan dans les datasets
def adjust_noise_level(signal, target_snr):
    current_snr = calculate_snr(signal)
    adjustment = (current_snr - target_snr) / 10.0
    adjusted_signal = signal + adjustment * np.random.normal(0, 1, signal.shape)  
    return adjusted_signal

#Normalizer le signal
def normalize_signal(matrix):
    matrix = matrix.astype(float)
    scaler = StandardScaler()
    norm_matrix = np.empty_like(matrix)
    for i in range(matrix.shape[0]):
        row = matrix[i, :]
        norm_row = scaler.fit_transform(row.reshape(-1, 1))
        norm_matrix[i, :] = norm_row.flatten()
    return norm_matrix


#Ajouter le noise au hdf5
def append_noise_data_to_hdf5(path, new_data, group_name='data'):
    with h5py.File(path, 'a') as hdf5:
        if group_name not in hdf5:
            hdf5.create_group(group_name)
        group = hdf5[group_name]
        for dataset_name, data_array in new_data.items():
            if dataset_name in group:
                del group[dataset_name]
            group.create_dataset(dataset_name, data=data_array , compression='gzip', shuffle=True)
            
"""            
#Merger des csv
def merge_csv_files(file1_path, file2_path, merged_file_path):
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    merged_df = pd.concat([df1, df2], axis=1)
    merged_df.to_csv(merged_file_path, index=False)
merge_csv_files("Metadata.csv", "test.csv", "mergetest.csv")
"""
# Define your variables
earthquake = 'C:/Users/alain/Bureau/Projet Yan/earthquake-main/Data/events_counts.hdf5'
noise = 'C:/Users/alain/Bureau/Projet Yan/earthquake-main/Data/Instance_noise.hdf5'
debutTemps = 0
finTemps = 12000
chunkSize = 36000
sampleSize = 1000
nom_fichier_output = "test.hdf5"
target_SNR = 5
nombre_noise = 1
hdf_target = noise

#hdf5_creation(hdf_target, debutTemps, finTemps, chunkSize, sampleSize, target_SNR)










    
