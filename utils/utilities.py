import mlflow
import skopt

from utils.training import train_validate

from config import MODEL


def push_mlflow(self, hyperparams, results, save_model=False, tags=None):
    with mlflow.start_run(experiment_id=self.exp_id) as active_run:

        for hyperparam, value in hyperparams.items():
            mlflow.log_param(hyperparam, value)

        for metric, value in results.items():
            mlflow.log_metric(metric, value)

        mlflow.set_tags(tags)

        if save_model:
            print('\nLogging model in MLflow:')
            mlflow.sklearn.log_model(self.clf, MODEL)



