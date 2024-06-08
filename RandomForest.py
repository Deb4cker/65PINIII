from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib


def __init__(self):
    #Inicialização da classe
    pass


def initTrain(X_train, y_train, X_validation, y_validation, X_test, y_test):
    param_grid_dt = {
        'criterion': ['gini', 'entropy'],
        'n_estimators': [50],
        'max_depth': [None, 5, 9, 10, 11, 12, 15, 20],
        'min_samples_split': [2, 3, 4, 5, 10, 15],
        'min_samples_leaf': [2, 3, 4, 5, 6, 10]
    }

    #Pegando os melhores parâmentros
    grid_search_dt = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=param_grid_dt,
                                        n_iter=500,
                                        cv=5, scoring='accuracy', n_jobs=-1)
    grid_search_dt.fit(X_train, y_train)

    print("Grid for the Random Forest")
    print(" ")
    print("Best Hyperparameters:")
    print(grid_search_dt.best_params_)
    print(" ")
    print("Performance on the Validation Set:")
    print(grid_search_dt.best_score_)
    print(" ")

    # Inicializar e treinar o modelo RandomForest
    fruits_random_forest_model = RandomForestClassifier(random_state=1)
    fruits_random_forest_model.fit(X_train, y_train)

    # Fazer previsões com o conjunto de validação e calcular a acurácia
    rf_val_predictions = fruits_random_forest_model.predict(X_validation)
    rf_val_accuracy = accuracy_score(y_validation, rf_val_predictions)
    print("Acurácia do modelo RandomForest no conjunto de validação:", rf_val_accuracy)

    # Fazer previsões com o conjunto de teste e calcular a acurácia
    rf_test_predictions = fruits_random_forest_model.predict(X_test)
    rf_test_accuracy = accuracy_score(y_test, rf_test_predictions)
    print("Acurácia do modelo RandomForest no conjunto de teste:", rf_test_accuracy)
    rf_f1_score = f1_score(y_validation, rf_val_predictions, average='weighted')
    print("F1-score do modelo RandomForest no conjunto de validação:", rf_f1_score)

    joblib.dump(fruits_random_forest_model, 'models/random_forest_model.pkl')
