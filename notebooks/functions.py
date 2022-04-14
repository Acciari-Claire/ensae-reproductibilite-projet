"""Functions used to import and prepare data before running a random forest."""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


def import_data(prefix):
    """ Import the Titanic training and testing sets """
    return pd.read_csv(prefix+"train.csv"), pd.read_csv(prefix+"test.csv")


def extract_title(dataframe):
    """ Extract the title from the name variable """
    title = dataframe['Name'].str.rsplit(",", n=1).str[-1]
    title = title.str.split().str[0]
    return title


def feature_engineering(dataframe, mean_age):
    """ Performs feature engineering steps """
    dataframe["Title"] = extract_title(dataframe)
    dataframe['Title'] = dataframe['Title'].replace('Dona.', 'Mrs.')
    dataframe['Age'] = dataframe['Age'].fillna(mean_age)
    dataframe['Ticket_Len'] = dataframe['Ticket'].apply(lambda x: len(x))
    dataframe['Fare'] = dataframe['Fare'].fillna(dataframe['Fare'].mean())
    dataframe['hasCabin'] = dataframe.Cabin.notnull().astype(int)
    dataframe['Embarked'] = dataframe['Embarked'].fillna('S')
    dataframe.drop(labels=['PassengerId', 'Name',
                           'Ticket', 'Cabin'], axis=1, inplace=True)
    return dataframe


def label_encoding(dataframe, var):
    """ Tranforms labels into numbers """
    encoder = LabelEncoder()
    dataframe[var] = encoder.fit_transform(dataframe[var].values)
    return dataframe


def split_train_test(dataframe):
    """ Splits the data in training and testing sets """
    y = dataframe["Survived"].values
    X = dataframe.drop(labels="Survived", axis=1).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    return X_train, X_test, y_train, y_test


def rf_train_evaluate(X_train, y_train, X_test, y_test, n_estimators):
    """ Runs and evaluate a random Forest Classifier """
    rdmf = RandomForestClassifier(n_estimators=n_estimators)
    rdmf.fit(X_train, y_train)

    rdmf_score = rdmf.score(X_test, y_test)
    print("{} % de bonnes réponses sur les données de test pour validation (résultat qu'on attendrait si on soumettait notre prédiction sur le dataset de test.csv)".format(round(rdmf_score*100)))
    print("matrice de confusion")
    print(confusion_matrix(y_test, rdmf.predict(X_test)))
