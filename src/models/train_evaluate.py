""" Function used to train a Random Forest Classifier and evaluate it """
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


def rf_train_evaluate(X_train, y_train, X_test, y_test, n_estimators):
    """ Runs and evaluate a random Forest Classifier """
    rdmf = RandomForestClassifier(n_estimators=n_estimators)
    rdmf.fit(X_train, y_train)

    rdmf_score = rdmf.score(X_test, y_test)
    print("{} % de bonnes réponses sur les données de test pour validation (résultat qu'on attendrait si on soumettait notre prédiction sur le dataset de test.csv)".format(round(rdmf_score*100)))
    print("matrice de confusion")
    print(confusion_matrix(y_test, rdmf.predict(X_test)))
