"""Runs a Random Forest on Titanic survival dataset."""
from src.data.import_dataset import import_data
from src.data.split_dataset import split_train_test
from src.features.build_features import feature_engineering, label_encoding
from src.models.train_evaluate import rf_train_evaluate


if __name__ == "__main__":

    TrainingData, TestData = import_data("")

    meanAge = round(TrainingData['Age'].mean())
    TrainingData = feature_engineering(TrainingData, meanAge)
    TestData = feature_engineering(TestData, meanAge)

    TrainingData = label_encoding(TrainingData, "Sex")
    TrainingData = label_encoding(TrainingData, "Title")
    TrainingData = label_encoding(TrainingData, "Embarked")

    X_train, X_test, y_train, y_test = split_train_test(TrainingData)

    rf_train_evaluate(X_train, y_train, X_test, y_test, 20)
