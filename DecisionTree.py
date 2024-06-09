from sklearn.model_selection import GridSearchCV
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
    grid_search_dt = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grid_dt, refit=True,
                                  cv=5, scoring='accuracy', n_jobs=-1)
    grid_search_dt.fit(X_train, y_train)

    print("-----------Decision Tree-----------")
    print("Grid for the Decision Tree")
    print("Best Hyperparameters: %s", grid_search_dt.best_params_)
    print("Performance on the Validation Set: %s", grid_search_dt.best_score_)

    # Inicializar e treinar o modelo DecisionTree
    fruits_decision_tree_model = DecisionTreeClassifier(random_state=1)
    fruits_decision_tree_model.fit(X_train, y_train)

    # Fazer previsões com o conjunto de validação e calcular a acuráciae f1_score
    dt_val_predictions = fruits_decision_tree_model.predict(X_validation)
    dt_val_accuracy = accuracy_score(y_validation, dt_val_predictions)
    dt_f1_score = f1_score(y_validation, dt_val_predictions, average='weighted')
    print("Validation Set Performance:")
    print("Accuracy: %s", dt_val_accuracy)
    print("F1-score: %s", dt_f1_score)

    # Fazer previsões com o conjunto de teste e calcular a acurácia e f1_score
    dt_test_predictions = fruits_decision_tree_model.predict(X_test)
    dt_test_accuracy = accuracy_score(y_test, dt_test_predictions)
    dt_test_f1_score = f1_score(y_test, dt_test_predictions, average='weighted')
    print("Validation Set Performance Decision Tree:")
    print("Test Set Decision Tree Accuracy: %s", dt_test_accuracy)
    print("Test Set Decision Tree f1 Score: %s", dt_test_f1_score)
    print("-----------------------------------------------")
    joblib.dump(fruits_decision_tree_model, 'models/decision_tree_model.pkl')
