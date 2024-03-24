import mlflow
import skopt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils.features_engineering import assign_to_grid, scale, shap_feat_imp
from utils.training import train_validate, test
from utils.data_load import load_dataset
from utils.dataloaders.InstanceDataset import InstanceDataset

from config import \
    FEATURES, \
    HYPERPARAMETERS, \
    INSTANCE_EVENTS_FILENAME, \
    INSTANCE_NOISE_FILENAME, \
    INSTANCE_EVENTS_METADATA_FILENAME, \
    INSTANCE_NOISE_METADATA_FILENAME, \
    FEATURES_SCALING, \
    NB_ITER

'''
Make the hyperparameters tuning that optimize the features engineering and filtering and the hyperparameters of the model 
training. The objective function is the train_validate function that returns the score and the run_id. 
'''


def main():
    # ---------------------- Elisee ----------------------
    df_instance = InstanceDataset(INSTANCE_EVENTS_FILENAME, INSTANCE_EVENTS_METADATA_FILENAME,
                                 INSTANCE_NOISE_FILENAME, INSTANCE_NOISE_METADATA_FILENAME)
    # -------------------- Elisee END --------------------

    # ---------------------- Mathieu ----------------------
    # Load metadata CSVs
    df_events_meta = pd.read_csv(INSTANCE_EVENTS_METADATA_FILENAME, index_col='trace_name')
    df_noise_meta = pd.read_csv(INSTANCE_NOISE_METADATA_FILENAME, index_col='trace_name')
    df_noise_meta['source'] = [0] * len(df_noise_meta)
    df_events_meta['source'] = [1] * len(df_events_meta)

    df_events_meta = df_events_meta[df_noise_meta.columns]
    df_meta = pd.concat([df_events_meta, df_noise_meta])
    df_meta = df_meta.dropna(axis=1, how='all')

    # Data Enrich
    # Make feature lat-long grid
    lat_min = df_meta['station_latitude_deg'].min()
    lon_min = df_meta['station_longitude_deg'].min()
    df_meta['grid_cell'] = df_meta.apply(
        lambda x: assign_to_grid(x['station_latitude_deg'], x['station_longitude_deg'], lat_min, lon_min),
        axis=1
    )
    # Other data enrich...

    # Scaling between 0 and 1
    # Metadata

    # Signals

    # Scaling
    df_meta_scaled, train_scaler, outliers_scaler = scale(
        df_meta,
        FEATURES['numerical'].keys(),
        scaling_params=FEATURES_SCALING,
    )
    df_meta_scaled['source'] = df_meta['source']
    df_train = df_meta_scaled.sample(frac=0.8, random_state=42)
    df_test = df_meta_scaled.drop(df_train.index)


    # # features scaling
    # min_max_scaler = MinMaxScaler()
    # df_normalize = min_max_scaler.fit_transform(df_meta)


    # Join metadata with signals data
    #.join('trace_name')

    # Make SHAP features importance Analysis
    # shap_feat_imp()


    # ---------------------- Mathieu ----------------------
    # Hyperparams Tuning
    @skopt.utils.use_named_args(HYPERPARAMETERS)
    def objective(**hyperparams):
        metrc_eval = train_validate(hyperparams, df_train, df_test)
        if hyperparams['metric'] in ['auc', 'accuracy', 'f1', 'precision', 'recall']:
            return -1.0 * metrc_eval
        elif hyperparams['metric'] in ['log_loss', 'loss', 'mse', 'mae', 'rmse']:
            return metrc_eval

    results = skopt.gp_minimize(objective, dimensions=HYPERPARAMETERS, n_calls=NB_ITER)
    best_hyperparams = results.x

    # ---------------------- Philippe ----------------------
    # Scale test
    # test_data = scale(scaler=train_scaler)
    # test(test_data, best_hyperparams)
    # -------------------- Philippe END --------------------

    # Push results to MLflow
    # -------------------- Mathieu END --------------------



    # ---------------------- Yan/Yas/Elisee/Mathieu/Philippe ----------------------
    # Presentation des resultats
    # Plots
    # -------------------- Yan/Yas/Elisee/Mathieu/Philippe END --------------------


if __name__ == '__main__':
    main()
