import mlflow
import skopt
import pandas as pd

from utils.features_engineering import assign_to_grid
from utils.training import train_validate
from utils.data_load import load_dataset
from utils.dataloaders.InstanceDataset import InstanceDataset

from config import \
    HYPERPARAMETERS, \
    MODEL, \
    INSTANCE_EVENTS_FILENAME, \
    INSTANCE_NOISE_FILENAME, \
    INSTANCE_EVENTS_METADATA_FILENAME, \
    INSTANCE_NOISE_METADATA_FILENAME

'''
Make the hyperparameters tuning that optimize the features engineering and filtering and the hyperparameters of the model 
training. The objective function is the train_validate function that returns the score and the run_id. 
'''


def main():
    event_hdf5_file = ''
    event_metadata_file = ''
    noise_hdf5_file = ''
    noise_metadata_file = ''
    # TODO Normalize signals between 0 and 1?
    # df_noise = load_dataset(INSTANCE_NOISE_FILENAME)
    # df_events = load_dataset(INSTANCE_EVENTS_FILENAME)
    # df_instace = InstanceDataset(INSTANCE_EVENTS_FILENAME, INSTANCE_EVENTS_METADATA_FILENAME,
    #                              INSTANCE_NOISE_FILENAME, INSTANCE_NOISE_METADATA_FILENAME)

    # Load metadata CSVs
    df_events = pd.read_csv(INSTANCE_EVENTS_METADATA_FILENAME, index_col='trace_name')
    df_noise = pd.read_csv(INSTANCE_NOISE_METADATA_FILENAME, index_col='trace_name')
    df_events = df_events[df_noise.columns]
    df = pd.concat([df_events, df_noise])
    df = df.dropna(axis=1, how='all')

    # Make feature lat-long grid
    lat_min = df['station_latitude_deg'].min()
    lon_min = df['station_longitude_deg'].min()
    df['grid_cell'] = df.apply(
        lambda x: assign_to_grid(x['station_latitude_deg'], x['station_longitude_deg'], lat_min, lon_min),
        axis=1
    )

    # Make features scaling


    # Make SHAP features importance Analysis



    pass

    @skopt.utils.use_named_args(HYPERPARAMETERS)
    def objective(**hyperparams):
        # score, run_id = self.train_validate(hyperparams)
        score, run_id = train_validate(hyperparams)
        return -1.0 * score


if __name__ == '__main__':
    main()
