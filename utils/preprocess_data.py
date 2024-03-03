import pandas as pd


def load_data():
    '''
    TODO complete this
    :return:
    '''
    # Load data from file
    data = pd.read_csv('data.csv')
    return data