from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib


def __init__(self):
    #Inicialização da classe
   pass
def initTrain(X_train, y_train, X_validation, y_validation, X_test, y_test):

    param_grid_dt = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 5, 9, 10, 11, 12, 15, 20],
        'min_samples_split': [2, 3, 4, 5, 10, 15],
        'min_samples_leaf': [2, 3, 4, 5, 6, 10]
    }
    grid_search_dt = RandomizedSearchCV(estimator=DecisionTreeClassifier(), param_distributions=param_grid_dt,
                                        n_iter=500,
                                        cv=5, scoring='accuracy', n_jobs=-1)
    grid_search_dt.fit(X_train, y_train)

    print("Grid for the Decision Tree")
    print(" ")
    print("Best Hyperparameters:")
    print(grid_search_dt.best_params_)
    print(" ")
    print("Performance on the Validation Set:")
    print(grid_search_dt.best_score_)
    print(" ")

    # Inicializar e treinar o modelo DecisionTree
    fruits_decision_tree_model = DecisionTreeClassifier(random_state=1)
    fruits_decision_tree_model.fit(X_train, y_train)

    # Fazer previsões com o conjunto de validação e calcular a acurácia
    dt_val_predictions = fruits_decision_tree_model.predict(X_validation)
    dt_val_accuracy = accuracy_score(y_validation, dt_val_predictions)
    print("Acurácia do modelo DecisionTree no conjunto de validação:", dt_val_accuracy)

    # Fazer previsões com o conjunto de teste e calcular a acurácia
    dt_test_predictions = fruits_decision_tree_model.predict(X_test)
    dt_test_accuracy = accuracy_score(y_test, dt_test_predictions)
    print("Acurácia do modelo DecisionTree no conjunto de teste:", dt_test_accuracy)

    # Calcular o F1-score para as previsões do modelo DecisionTree
    dt_f1_score = f1_score(y_validation, dt_val_predictions, average='weighted')
    print("F1-score do modelo DecisionTree no conjunto de validação:", dt_f1_score)

    joblib.dump(fruits_decision_tree_model, 'models/decision_tree_model.pkl')