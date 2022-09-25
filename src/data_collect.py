import pandas as pd


def data_collect(train=True):

    if train:
        df = pd.read_csv('data/fraudTrain.csv')
    else:
        df = pd.read_csv('data/fraudTest.csv')
    
    return df