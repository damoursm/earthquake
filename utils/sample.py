# Ce fichier est un outil pour aider acceder et/ou extraire les données du csv et des fichier hdf5 pour faire des samples. 

# Pour avoir des info sur ce que signifies les variables du fichier csv tout est dispo dans le pdf de instance (page 10-11)
# Disponible au lien suivant :  https://essd.copernicus.org/preprints/essd-2021-164/essd-2021-164.pdf


#------------------------------------------------------------------------------Import pour le code------------------------------------------------------------------------------------------#

import pandas as pd
import numpy as np
import h5py

#------------------------------------------------------------------------------------Variable-----------------------------------------------------------------------------------------------#

# Path vers le fichier csv et hdf5 a sample sur votre ordinateur 
urlCsv = 'Data/metadata_Instance_events_v2.csv'
urlHdf5 = 'Data/Instance_events_counts_10k.hdf5'

#-----------------------------------Fonction----------------------------------------#


# Dans le hdf5 les signaux sont dans une subsection "data" et les code de chaque signal suit la nomenclature du
# fichier csv (source_id.station_network_code.station_code.station_location_code.station_channels).
# Les signaux sont de 120 seconde et 100hz.
# Ce sont des waveforms a 3 composants, il y a un composant nord-sud, un est-ouest et un vertical
# ce qui nous donne 3 * 12000 points par type de signal 
# si on veut un sample des 20 premiere seconde il faut donc prendre par exemple des 2000 premier points (100 points par secondes)

debutTemps = 0   #Le temps est calculer en centieme de seconde(100hz/s donc 1000 = 10s)
finTemps = 12000 # Max 12000  c'est un array (3:12000)
chunkSize = (36000) # Loader en chunk
sampleSize = 10 #Nombre de matrice voulu

#Code simple pour prendre les certains points des x premieres matrices d'un hdf5 en utilisant des chunks
def sample(urlHdf5, debutPoint ,maxPoint, chunkSize, sampleSize):
    with h5py.File(urlHdf5, 'r') as hdf5:
        with h5py.File('sample.hdf5', 'w') as sample:
            nomCategorie = 'data'                        # Structure du fichier hdf5 les info sont dans la section data ont peut acceder a d'autre partie en changeant le String ou placer ca dans un autre groupe
                                                         # Il serait possible de mettre toute les parties des hdf(noise, groundmotion et event) dans le meme fichier hdf5 si on veut dans des subsections differente.    
            categorieInput = hdf5[nomCategorie]
            categorieOutput = sample.create_group(nomCategorie)

            for i, nomDataset in enumerate(categorieInput):
                if i>= sampleSize:
                    return
                dataSet = categorieInput[nomDataset]
                rows, cols = dataSet.shape
                output_dataset = categorieOutput.create_dataset(nomDataset, shape=(rows, maxPoint - debutPoint))

                for start in range(0, rows, chunkSize):
                    end = min(start + chunkSize, rows) #Index de la fin du chunk
                    chunk = dataSet[start:end, debutPoint:maxPoint] #Extract les donnees du dataset
                    output_dataset[start:end, :] = chunk # write les donnee du chunk dans le output

#sample(urlHdf5,debutTemps,finTemps,chunkSize,sampleSize)


array = ['11030611.IV.OFFI..HH']
# Fonction pour get toute les matrices qui onrt le meme identifiant que dans le array 
# Il est possible de faire un filter pour juste get les id qui respecterait des conditions 
# predefinie dans le csv

def sampleArray(array, urlHdf5, debutPoint ,maxPoint, chunkSize):
    with h5py.File(urlHdf5, 'r') as hdf5:
        with h5py.File('sample.hdf5', 'w') as sample:
            nomCategorie = 'data'                        
                                                            
            categorieInput = hdf5[nomCategorie]
            categorieOutput = sample.create_group(nomCategorie)

            for i, nomDataset in enumerate(categorieInput):
                if i>= len(array):
                    return
                
                if nomDataset not in array:
                    continue
                dataSet = categorieInput[nomDataset]
                rows, cols = dataSet.shape
                output_dataset = categorieOutput.create_dataset(nomDataset, shape=(rows, maxPoint - debutPoint))

                for start in range(0, rows, chunkSize):
                    end = min(start + chunkSize, rows) #Index de la fin du chunk
                    chunk = dataSet[start:end, debutPoint:maxPoint] #Extract les donnees du dataset
                    output_dataset[start:end, :] = chunk # write les donnee du chunk dans le output

sampleArray(array, urlHdf5,debutTemps,finTemps,chunkSize)


#def arrayCondition():
#def randomSample():




























