import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


def import_data():
    return pd.read_csv("train.csv"), pd.read_csv("test.csv")


def extract_title(df):
    x = df['Name'].apply(lambda x: x.split(',')[1])
    x = x.apply(lambda x: x.split()[0])
    return x


def feature_engineering(df, meanAge):
    df["Title"] = extract_title(df)
    df['Title'] = df['Title'].replace('Dona.', 'Mrs.')
    df['Age'] = df['Age'].fillna(meanAge)
    df['Ticket_Len'] = df['Ticket'].apply(lambda x: len(x))
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    df['hasCabin'] = df.Cabin.notnull().astype(int)
    df['Embarked'] = df['Embarked'].fillna('S')
    df.drop(labels=['PassengerId', 'Name',
                    'Ticket', 'Cabin'], axis=1, inplace=True)
    return df


def label_encoding(df, var):
    encoder = LabelEncoder()
    df[var] = encoder.fit_transform(df[var].values)
    return df


def split_train_test(df):
    y = df["Survived"].values
    X = df.drop(labels="Survived", axis=1).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    return X_train, X_test, y_train, y_test


def RF_train_evaluate(X_train, y_train, X_test, y_test, n_estimators):
    rdmf = RandomForestClassifier(n_estimators=20)
    rdmf.fit(X_train, y_train)

    rdmf_score = rdmf.score(X_test, y_test)
    print("{} % de bonnes réponses sur les données de test pour validation (résultat qu'on attendrait si on soumettait notre prédiction sur le dataset de test.csv)".format(round(rdmf_score*100)))
    print("matrice de confusion")
    print(confusion_matrix(y_test, rdmf.predict(X_test)))
