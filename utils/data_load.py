from __future__ import print_function
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from matplotlib import pyplot as plt


def load_dataset(filename):
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
                hdf5_data = []
                trace_names = []
                for key in data.keys():
                    dataset = data[key]
                    if isinstance(dataset, h5py.Dataset):
                        # Vérifier la forme des données extraites
                        if dataset.ndim == 2:
                            # Concaténer les trois canaux pour chaque signal
                            concatenated_data = np.concatenate([np.atleast_2d(dataset[i, :]) for i in range(3)], axis=1)
                            hdf5_data.append(concatenated_data)
                            trace_names.append(key)
                        else:
                            print(f"Les données pour la clé '{key}' ne sont pas sous forme de tableau 2D.")
                if hdf5_data:
                    stacked_data_matrix = np.vstack(hdf5_data)
                    df_hdf5 = pd.DataFrame(stacked_data_matrix)
                    df_hdf5['trace_name'] = trace_names
                    return df_hdf5


################    récuperation des nom de colonnes en commun dans le csv de noise et earthquake
def  common_columns(csv_file1,csv_file2):
    earth_csv = pd.read_csv(csv_file1)
    noise_csv = pd.read_csv(csv_file2)
    # Obtenir les noms de colonnes en commun
    common_columns = earth_csv.columns.intersection(noise_csv.columns)
    # Afficher les noms de colonnes en commun
    print(common_columns)


if __name__ == '__main__':
    # Utilisation de la fonction pour charger et convertir le dataset à partir du fichier HDF5
    filename1 = 'Instance_sample_dataset/data/Instance_noise_1k.hdf5'
    df_noise = load_dataset(filename1)
    # Ajouter une colonne 'source_type' avec des valeurs constantes de 1
    df_noise['source_type'] = 1

    filename2 = 'Instance_sample_dataset/data/Instance_events_counts_10k.hdf5'
    df_earth = load_dataset(filename2)
    len(df_earth)
    df_earth['source_type'] = 0
    print(df_earth)

    ############################    récuperation des données dans les csv
    # Spécifier le chemin complet vers le fichier CSV
    csv_file1 = 'Instance_sample_dataset/metadata/metadata_Instance_noise_1k.csv'
    # Lecture du fichier CSV contenant les informations sur la base de données des signaux
    noise_csv = pd.read_csv(csv_file1)

    # Spécifier le chemin complet vers le fichier CSV
    csv_file2 = 'Instance_sample_dataset/metadata/metadata_Instance_events_10k.csv'
    # Lecture du fichier CSV contenant les informations sur la base de données des signaux
    earth_csv = pd.read_csv(csv_file2)
    print(earth_csv)

    # Spécifier le chemin complet vers le fichier CSV
    csv_file1 = 'Instance_sample_dataset/metadata/metadata_Instance_events_10k.csv'
    # Spécifier le chemin complet vers le fichier CSV
    csv_file2 = 'Instance_sample_dataset/metadata/metadata_Instance_noise_1k.csv'

    common_columns(csv_file1, csv_file2)
