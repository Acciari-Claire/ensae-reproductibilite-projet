""" Functions used to build the features and encode their labels """
from sklearn.preprocessing import LabelEncoder


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
