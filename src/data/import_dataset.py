"""Function used to import Titanic training and testing datasets."""
import pandas as pd


def import_data(prefix):
    """ Import the Titanic training and testing sets """
    return pd.read_csv(prefix+"data/raw/train.csv"), pd.read_csv(prefix+"data/raw/test.csv")
