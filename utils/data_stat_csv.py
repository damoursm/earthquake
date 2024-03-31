# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:34:22 2024
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import h5py


def merge_csv_correl(noiseCSV, earthquakeCSV, output_file):
    df1 = pd.read_csv(noiseCSV)
    df2 = pd.read_csv(earthquakeCSV)
    df1['source'] = 0
    df2['source'] = 1
    df2 = df2[df1.columns]
    merged_df = pd.concat([df1, df2], axis=0)
    merged_df.reset_index(drop=True, inplace=True)
    merged_df.to_csv(output_file, index=False)
#merge_csv_correl("path1", "path2", "path3") Il faut mettre les path vers les fichiers


def matrixCorrelation(df):
    columnExcluded = ['source_id','station_network_code','trace_dt_s','trace_npts','station_code','station_location_code','station_channels','station_vs_30_detail','trace_start_time','trace_name']
    correlSubset = df.drop(columnExcluded, axis=1)
    correl = correlSubset.corr()
    plt.figure(figsize=(40, 40))
    sns.heatmap(correl, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()
    
#matrixCorrelation('pd.read_csv(PATH VERS LE CSV MERGÉ)')

def stat_hdf5(urlHdf5, output_csv='test.csv'):
    stats_df = pd.DataFrame(columns=["mean_E_Norma", "median_E_Norma", "max_E_Norma", "min_E_Norma", "rms_E_Norma","lquartile_E_Norma", "uquartile_E_Norma",
                                     "mean_N_Norma", "median_N_Norma", "max_N_Norma", "min_N_Norma", "rms_N_Norma","lquartile_N_Norma", "uquartile_N_Norma",
                                     "mean_Z_Norma", "median_Z_Norma", "max_Z_Norma", "min_Z_Norma", "rms_Z_Norma", "lquartile_Z_Norma", "uquartile_Z_Norma"])
    
    with h5py.File(urlHdf5, 'r') as hdf5:
        for i, nom_dataset in enumerate(hdf5['data']):
            data_set = hdf5['data'][nom_dataset]
            noise_data = process_data(data_set)
            stats = calculate_stats(noise_data)
            stats_df.loc[i] = stats
        
    stats_df.to_csv(output_csv, index=False)

def process_data(data_set):
    rows, cols = data_set.shape
    noise_data = np.empty((rows, cols))
    for i in range(rows):
        row = data_set[i]
        noise_data[i] = row
    return noise_data
    
def calculate_stats(noise_data):
    stats = {}
    stats["mean_E_Norma"] = np.mean(noise_data[0], axis=0)
    stats["median_E_Norma"] = np.median(noise_data[0],axis=0)
    stats["max_E_Norma"] = np.max(noise_data[0],axis=0)
    stats["min_E_Norma"] = np.min(noise_data[0],axis=0)
    stats["rms_E_Norma"] = np.sqrt(np.mean(noise_data[0]**2, axis=0))#root mean square
    stats["lquartile_E_Norma"] = np.percentile(noise_data[0], 25) #lower quartile
    stats["uquartile_E_Norma"] = np.percentile(noise_data[0], 75) #upper quartile
    
    stats["mean_N_Norma"] = np.mean(noise_data[1],axis=0)
    stats["median_N_Norma"] = np.median(noise_data[1],axis=0)
    stats["max_N_Norma"] = np.max(noise_data[1],axis=0)
    stats["min_N_Norma"] = np.min(noise_data[1],axis=0)
    stats["rms_N_Norma"] = np.sqrt(np.mean(noise_data[1]**2, axis=0)) 
    stats["lquartile_N_Norma"] = np.percentile(noise_data[0], 25) #lower quartile
    stats["uquartile_N_Norma"] = np.percentile(noise_data[0], 75) #upper quartile
    
    stats["mean_Z_Norma"] = np.mean(noise_data[2],axis=0)
    stats["median_Z_Norma"] = np.median(noise_data[2],axis=0)
    stats["max_Z_Norma"] = np.max(noise_data[2],axis=0)
    stats["min_Z_Norma"] = np.min(noise_data[2],axis=0)
    stats["rms_Z_Norma"] = np.sqrt(np.mean(noise_data[2]**2, axis=0)) 
    stats["lquartile_Z_Norma"] = np.percentile(noise_data[0], 25) #lower quartile
    stats["uquartile_Z_Norma"] = np.percentile(noise_data[0], 75) #upper quartile
    return stats

        
#Merger des csv
def merge_csv_files(file1_path, file2_path, merged_file_path):
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    merged_df = pd.concat([df1, df2], axis=1)
    merged_df.to_csv(merged_file_path, index=False)
#merge_csv_files("Metadata.csv", "test.csv", "mergetest.csv")

def commonColumn(pathE, pathN):
    df1 = pd.read_csv(pathE)
    df2 = pd.read_csv(pathN)
    common_columns = [col for col in df1.columns if col in df2.columns]
    df2 = df2[common_columns]
    df1_common = df1.reindex(columns=common_columns)
    df2_common = df2.reindex(columns=common_columns)
    df1_common.to_csv('Earthquake_common.csv', index=False)
    df2_common.to_csv('Noise_common.csv', index=False)
    
#stat_hdf5('20kNoiseAugmente.hdf5', 'Noise_common.csv')
