"""Functions used to split the dataset."""
from sklearn.model_selection import train_test_split


def split_train_test(dataframe):
    """ Splits the data in training and testing sets """
    y = dataframe["Survived"].values
    X = dataframe.drop(labels="Survived", axis=1).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    return X_train, X_test, y_train, y_test
