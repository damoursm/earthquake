import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import h5py
from sklearn.preprocessing import StandardScaler



"""
Ce code sert principalement a normalizer les donnees et ajouter du bruit gaussien en utilisant principes
de SNR. Il est /galement possible de couper les signaux pour prendre justes certain intervalles ou merger les noises et earthaquake dans le 
meme hdf5 en runnant la fonction avec HDF_TARGET = Noise et HDF_TARGET = Earthquake. La fonction creer un nouveau fichier HDF5 avec ce qui a ete specifier.

****   SI LE FICHIER EXISTE DEJA LA FONCTION APPEND A LA PLACE DONC NE PAS RUN SUR LE DATASET COMPLET POUR EVITER DE MODIFIER LA DATABASE  ****
NomFichierOutput = "NE_PAS_CHOISIR_LA_DATA_BASE.hdf5"     

 J'ai egalement inclu une fonction pour merges les csv ensembles si vous en avez besoin et le code pour faire
la matrice de correlation si jamais on s'en sert
"""



def merge_csv(noiseCSV, earthquakeCSV, output_file):
    df1 = pd.read_csv(noiseCSV)
    df2 = pd.read_csv(earthquakeCSV)
    df1['source'] = 0
    df2['source'] = 1
    df2 = df2[df1.columns]
    merged_df = pd.concat([df1, df2], axis=0)
    merged_df.reset_index(drop=True, inplace=True)
    merged_df.to_csv(output_file, index=False)
#merge_csv("path1", "path2", "path3") Il faut mettre les path vers les fichiers


df ='pd.read_csv(PATH VERS LE CSV MERGÉ)'
def matrixCorrelation():
    columnExcluded = ['source_id','station_network_code','trace_dt_s','trace_npts','station_code','station_location_code','station_channels','station_vs_30_detail','trace_start_time','trace_name']
    correlSubset = df.drop(columnExcluded, axis=1)
    correl = correlSubset.corr()
    plt.figure(figsize=(40, 40))
    sns.heatmap(correl, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()
#matrixCorrelation()


#<------------------------------------------------------------------###### PARAMETRE FONCTION #######------------------------------------------------------->
#Mettre vos path pour les fichier HDF5
Earthquake = ''
Noise = ''
debutTemps = 0   #Le temps est calculer en centieme de seconde(100hz/s donc 1000 = 10s) ça donne le premier point x qu'on veut
finTemps = 12000 # Max 12000 il y a 12000 x dans la matrice
chunkSize = 36000 # loader en chunk pour eviter probleme de memoire
sampleSize = 5 # nombre de earthquake ou noise qu'on veut dans le sample
NomFichierOutput = "exemple.hdf5" # SUPER IMPORTANT DE NE PAS CHOISIR NOISE OU EARTHQUAKE IL FAUT CREER UN NOUVEAU FICHIER
target_SNR = 8 #SNR voulu
HDF_TARGET = Noise #Noise ou Earthquake


# Dans le hdf5 les signaux sont dans une subsection "data" et les code de chaque signal suit la nomenclature du
# fichier csv (source_id,station_network_code,station_code,station_location_code,station_channels).
# Les signaux sont de 120 seconde et 100hz.
# # composant ce qui nous donne 3 * 12000 points par earthquake 
# si on veut un sample des 20 premiere seconde il faut donc prendre par exemple des 2000 premier points (100 points par secondes)

def hdf5Creation(urlEventHdf5, debutPoint, maxPoint, chunkSize, sampleSize):
    if HDF_TARGET == NomFichierOutput : return 
    with h5py.File(urlEventHdf5, 'r') as hdf5:
        new_noise_data = {}
        for i, nomDataset in enumerate(hdf5['data']):
            if i >= sampleSize: #Pour si on veut des plus petit samples
                break
            dataSet = hdf5['data'][nomDataset]
            rows, cols = dataSet.shape
            new_noise_data = {}  # Nouveau dictionary pour chaque dataset
            if urlEventHdf5==Earthquake:
                for start in range(0, rows, chunkSize):
                    end = min(start + chunkSize, rows)
                    chunk = dataSet[start:end, debutPoint:maxPoint]
                    chunk = normalizeSignal(chunk)
                    
                    # Adjuster noise level selon le SNR
                    adjusted_chunk = adjust_noise_level(chunk, target_SNR)
                    
                    noise_data = np.empty((rows, maxPoint - debutPoint))
                    noise_data[start:end, :] = adjusted_chunk 
                new_noise_data[f"{nomDataset}_earthquake"] = noise_data
                
            if urlEventHdf5==Noise:
                for j in range(5): #Permet dajuster le nombre de copie de chaque noise data ave un gaussian differents
                    noise_data = np.empty((rows, maxPoint - debutPoint))  # New noise_data array for each loop
                    for start in range(0, rows, chunkSize):
                        end = min(start + chunkSize, rows)
                        chunk = dataSet[start:end, debutPoint:maxPoint]
                        chunk = normalizeSignal(chunk)
                        # Adjust noise level based on target SNR
                        adjusted_chunk = adjust_noise_level(chunk, target_SNR)
                        
                        noise_data = np.empty((rows, maxPoint - debutPoint))
                        noise_data[start:end, :] = adjusted_chunk
                        
                    new_noise_data[f"{nomDataset}_noise_{j}"] = noise_data  # Renaming the dataset
            appendNoiseDataToHDF5(NomFichierOutput, new_noise_data)

#Calculer le snr pour definir la quantite de bruit
#https://en.wikipedia.org/wiki/Signal-to-noise_ratio 
def calculate_snr(matrix):

    snr_values = []
    for row in matrix:
        signal_power = np.sum(row ** 2)  # Power du signal
        noise_power = np.sum((row - np.mean(row)) ** 2)  # Power du noise
        snr = 10 * np.log10(signal_power / noise_power)  # Calculer SNR 
        snr_values.append(snr)
    avg_snr = np.mean(snr_values)  # SNR moyen
    return avg_snr
    
def adjust_noise_level(signal, target_snr):
    current_snr = calculate_snr(signal)
    adjustment = (current_snr - target_snr) / 10.0
    adjusted_signal = signal + adjustment * np.random.normal(0, 1, signal.shape)  
    return adjusted_signal


#Fonction pour normalizer les signaux row par row
def normalizeSignal(matrix):
    matrix = matrix.astype(float)  # Converter en float
    scaler = StandardScaler()
    normMatrix = np.empty_like(matrix)
    for i in range(matrix.shape[0]):
        row = matrix[i, :]
        normRow = scaler.fit_transform(row.reshape(-1, 1))
        normMatrix[i, :] = normRow.flatten()
    return normMatrix

"""
#Fonction pour ajouter du bruit gaussien a une matrice
def addGaussianNoise(matrix, mean = 0, std = 1):
    noise = np.random.normal(mean, std, size = matrix.shape)
    matrix = matrix+noise
    return matrix
"""
#La fonction sert a ajouter des entry au hdf5
def appendNoiseDataToHDF5(path, new_data, group_name='data'):
    with h5py.File(path, 'a') as hdf5:
        if group_name not in hdf5:
            hdf5.create_group(group_name)
        group = hdf5[group_name]
        for dataset_name, data_array in new_data.items():
            if dataset_name in group:
                del group[dataset_name]  
            group.create_dataset(dataset_name, data=data_array)




hdf5Creation(HDF_TARGET, debutTemps, finTemps, chunkSize, sampleSize)










    