# -*- coding: utf-8 -*-
#import pandas as pd
import numpy as np
import h5py
import pandas as pd
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
                new_noise_data[f"{nom_dataset}"] = noise_data
            elif url_event_hdf5 == noise:
                noise_data = process_data(data_set, debut_point, max_point, chunk_size, target_snr, nombre_noise)
                new_noise_data[f"{nom_dataset}"] = noise_data
            append_noise_data_to_hdf5(nom_fichier_output, new_noise_data)

            
            
def process_data(data_set, debut_point, max_point, chunk_size, target_snr, nombre_noise=1):
    rows, cols = data_set.shape
    noise_data = np.empty((rows, max_point - debut_point))
    for j in range(nombre_noise):
        for start in range(0, rows, chunk_size):
            end = min(start + chunk_size, rows)
            chunk = data_set[start:end, debut_point:max_point]
            chunk = normalize_signal(chunk)
            #adjusted_chunk = adjust_noise_level(chunk, target_snr)
            noise_data[start:end, :] = chunk
    return noise_data


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
            
            
def remove_suffix(dataset_name):
    if dataset_name.endswith('_earthquake'): #sUFFIX A ENLEVER
        return dataset_name[:-len('_earthquake')] #sUFFIX A ENLEVER
    return dataset_name


def removeSuffix(hdf5_file_path):
    with h5py.File(hdf5_file_path, 'r+') as hdf5_file:
        datasets = [str(dataset) for dataset in list(hdf5_file['data'])]
        for dataset_name in datasets:
            new_name = remove_suffix(dataset_name)
            hdf5_file['data'].move(dataset_name, new_name)

removeSuffix('earthquake.hdf5')

def sampleCSV(hdf5_file_path, csv_file_path, output_csv_path):
    with h5py.File(hdf5_file_path, 'r+') as hdf5_file:
        datasets = list(hdf5_file['data'])
    df = pd.read_csv(csv_file_path)
    entry_columns = ['source_id', 'station_network_code', 'station_code', 'station_location_code', 'station_channels']  
    entry = df[entry_columns].apply(lambda x: '.'.join(x.astype(str)), axis=1).str.replace('nan', '')
    filtered_df = df[entry.isin(datasets)]
    filtered_df.to_csv(output_csv_path, index=False)
    
    num_deleted_entries = removeNotCSV('noise.hdf5', datasets)
    print("Number of deleted entries:", num_deleted_entries)

#Fonction pour enlever dans le hdf les entry qui ne sont pas dans le csv
def removeNotCSV(hdf5_file_path, datasets_to_keep):
    num_deleted_entries = 0
    with h5py.File(hdf5_file_path, 'r+') as hdf5_file:
        datasets = [str(dataset) for dataset in list(hdf5_file['data'])]  
        for dataset_name in datasets:
            if dataset_name not in datasets_to_keep:
                del hdf5_file['data'][dataset_name]
                num_deleted_entries += 1
    return num_deleted_entries
#sampleCSV('noise.hdf5', 'metadata_Instance_noise.csv', 'noise.csv')


#Fonction pour enlever au csv les entry qui n'ont pas de pahse p ou s
def removeNoPhaseP_S(csv_file_path, csv_output):
    df = pd.read_csv(csv_file_path)
    empty_rows = df['trace_P_arrival_sample'].isna() & df['trace_S_arrival_sample'].isna()
    df = df[~empty_rows]
    df.to_csv(csv_output, index=False)

#removeNoPhaseP_S('earthquake.csv', 'test.csv')


earthquake = 'earthquake.hdf5'
noise = 'noise.hdf5'
debutTemps = 0
finTemps = 12000
chunkSize = 36000
sampleSize = 20
nom_fichier_output = "test.hdf5"
target_SNR = 5
nombre_noise = 1
hdf_target = earthquake


#hdf5_creation(hdf_target, debutTemps, finTemps, chunkSize, sampleSize, target_SNR)









    
