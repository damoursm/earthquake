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
    stats_df = pd.DataFrame(columns=["mean_E", "median_E", "max_E", "min_E", "rms_E","lquartile_E", "uquartile_E",
                                     "mean_N", "median_N", "max_N", "min_N", "rms_N","lquartile_N", "uquartile_N",
                                     "mean_Z", "median_Z", "max_Z", "min_Z", "rms_Z", "lquartile_Z", "uquartile_Z"])
    
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
    stats["mean_E"] = np.mean(noise_data[0], axis=0)
    stats["median_E"] = np.median(noise_data[0],axis=0)
    stats["max_E"] = np.max(noise_data[0],axis=0)
    stats["min_E"] = np.min(noise_data[0],axis=0)
    stats["rms_E"] = np.sqrt(np.mean(noise_data[0]**2, axis=0))#root mean square
    stats["lquartile_E"] = np.percentile(noise_data[0], 25) #lower quartile
    stats["uquartile_E"] = np.percentile(noise_data[0], 75) #upper quartile
    
    stats["mean_N"] = np.mean(noise_data[1],axis=0)
    stats["median_N"] = np.median(noise_data[1],axis=0)
    stats["max_N"] = np.max(noise_data[1],axis=0)
    stats["min_N"] = np.min(noise_data[1],axis=0)
    stats["rms_N"] = np.sqrt(np.mean(noise_data[1]**2, axis=0)) 
    stats["lquartile_N"] = np.percentile(noise_data[0], 25) #lower quartile
    stats["uquartile_N"] = np.percentile(noise_data[0], 75) #upper quartile
    
    stats["mean_Z"] = np.mean(noise_data[2],axis=0)
    stats["median_Z"] = np.median(noise_data[2],axis=0)
    stats["max_Z"] = np.max(noise_data[2],axis=0)
    stats["min_Z"] = np.min(noise_data[2],axis=0)
    stats["rms_Z"] = np.sqrt(np.mean(noise_data[2]**2, axis=0)) 
    stats["lquartile_Z"] = np.percentile(noise_data[0], 25) #lower quartile
    stats["uquartile_Z"] = np.percentile(noise_data[0], 75) #upper quartile
    return stats

stat_hdf5('test.hdf5', 'test.csv')