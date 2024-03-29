from datetime import datetime
import mlflow
import os
import skopt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils.features_engineering import data_enrich, scale, shap_feat_imp
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


class MLflowExperiment():
    def __init__(self):
        self.exp_id = None
        self.run_id = None
        self.model = None
        # self.best_score = None
        # self.best_params = None
        # self.best_model = None
        # self.best_features = None
        # self.best_run_id = None
        # self.best_model_name = None
        # self.best_model_type = None
        # self.best_model_features = None
        # self.best_model_hyperparams = None
        # self.best_model_score = None
        # self.best_model_shap_values = None
        # self.best_model_shap_values_abs = None
        self.df_train = None
        self.df_test = None

        # Setup MLflow experiment
        self.client = mlflow.tracking.MlflowClient()
        now = datetime.now()  # current date and time
        date_time = now.strftime("%Y-%m-%d %H:%M:%S")
        self.exp_id = self.client.create_experiment(f"Earthquake: {date_time}")

    def features_engineering(self):
        # Load metadata CSVs
        df_events_meta = pd.read_csv(INSTANCE_EVENTS_METADATA_FILENAME, index_col='trace_name')
        df_noise_meta = pd.read_csv(INSTANCE_NOISE_METADATA_FILENAME, index_col='trace_name')
        df_noise_meta['source'] = [0] * len(df_noise_meta)
        df_events_meta['source'] = [1] * len(df_events_meta)

        df_events_meta = df_events_meta[df_noise_meta.columns]
        df_meta = pd.concat([df_events_meta, df_noise_meta])
        df_meta = df_meta.dropna(axis=1, how='all')

        # Data Enrich
        df_meta, ohe_df = data_enrich(df_meta)

        # Scaling between 0 and 1
        # Metadata

        # Enrich signals model data
        # df_meta = signal_enrich(df_meta)

        # Scaling metadata
        df_meta_scaled, train_scaler, outliers_scaler = scale(
            df_meta,
            FEATURES['numerical'].keys(),
            scaling_params=FEATURES_SCALING,
        )
        df_meta_scaled = pd.concat([df_meta_scaled, ohe_df], axis=1)
        global features_list
        features_list = list(df_meta_scaled.columns)
        df_meta_scaled['source'] = df_meta['source']

        # Split the dataset
        self.df_train = df_meta_scaled.sample(frac=0.8, random_state=42)
        self.df_test = df_meta_scaled.drop(self.df_train.index)

        # Make SHAP features importance Analysis
        # shap_feat_imp()

    def run_experiment(self):
        # Hyperparams Tuning
        @skopt.utils.use_named_args(HYPERPARAMETERS)
        def objective(**hyperparams):
            self.model, metrics, artifacts = train_validate(hyperparams, self.df_train, self.df_test)
            self.run_id = self.push_run(hyperparams, metrics, artifacts, tags={'test': False})
            if hyperparams['metric'] in ['auc', 'accuracy', 'f1', 'precision', 'recall']:
                return -1.0 * metrics['metric_eval']
            elif hyperparams['metric'] in ['log_loss', 'loss', 'mse', 'mae', 'rmse']:
                return metrics['metric_eval']

        results = skopt.gp_minimize(objective, dimensions=HYPERPARAMETERS, n_calls=NB_ITER)
        self.best_hyperparams = results.x

        # Scale test
        # test_data = scale(scaler=train_scaler)
        # test(test_data, best_hyperparams)

    def push_run(self, hyperparams, results, artifacts=[], save_model=False, tags=None):
        with mlflow.start_run(experiment_id=self.exp_id) as active_run:
            for hyperparam, value in hyperparams.items():
                mlflow.log_param(hyperparam, value)
            for metric, value in results.items():
                mlflow.log_metric(metric, value)
            for artifact in artifacts:
                mlflow.log_artifact(artifact)
            mlflow.set_tags(tags)

            if save_model:
                print('\nLogging model in MLflow:')
                mlflow.sklearn.log_model(self.model, hyperparams['name'])

        return active_run.info.run_id


def main():
    ml_exp = MLflowExperiment()
    ml_exp.features_engineering()
    ml_exp.run_experiment()


if __name__ == '__main__':
    main()
