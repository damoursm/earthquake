import mlflow
import skopt

from utils.training import train_validate

from config import HYPERPARAMETERS, MODEL


'''
Make the hyperparameters tuning that optimize the features engineering and filtering and the hyperparameters of the model 
training. The objective function is the train_validate function that returns the score and the run_id. 
'''


def main():
    @skopt.utils.use_named_args(HYPERPARAMETERS)
    def objective(**hyperparams):
        # score, run_id = self.train_validate(hyperparams)
        score, run_id = train_validate(hyperparams)
        return -1.0 * score


if __name__ == '__main__':
    main()
