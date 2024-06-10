from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging


def __init__():
    pass


def initTrain(X_train, y_train, X_validation, y_validation):
    param_grid_dt = {
        'criterion': ['gini', 'entropy'],
        'n_estimators': [10, 25, 35, 50],
        'max_depth': [None, 5, 9, 10, 11, 12, 15, 20],
        'min_samples_split': [2, 3, 4, 5, 10, 15],
        'min_samples_leaf': [2, 3, 4, 5, 6, 10]
    }

    #Pegando os melhores parâmentros
    grid_search_dt = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid_dt, refit=True,
                                  cv=5, scoring='accuracy', n_jobs=-1)
    grid_search_dt.fit(X_train, y_train)

    print("----------Random Forest Train-------------")
    print("Grid for the Random Forest")
    print("Best Hyperparameters: %s", grid_search_dt.best_params_)
    print("Performance on the Validation Set: %s", grid_search_dt.best_score_)

    # Inicializar e treinar o modelo RandomForest
    fruits_random_forest_model = RandomForestClassifier(random_state=1)
    fruits_random_forest_model.fit(X_train, y_train)

    # Fazer previsões com o conjunto de validação e calcular a acurácia
    rf_val_predictions = fruits_random_forest_model.predict(X_validation)
    rf_val_accuracy = accuracy_score(y_validation, rf_val_predictions)
    rf_f1_score = f1_score(y_validation, rf_val_predictions, average='weighted')

    print("Validation Set Performance:")
    print("Accuracy: %s", rf_val_accuracy)
    print("F1-score: %s", rf_f1_score)
    joblib.dump(fruits_random_forest_model, 'models/random_forest_model.pkl')


