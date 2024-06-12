from sklearn.metrics import accuracy_score, f1_score
import joblib


def __init__():
    pass


def init(X_test, y_test):
    # Substitua 'caminho_do_arquivo.pkl' pelo caminho real do seu arquivo .pkl
    file_path_random = 'models/random_forest_model.pkl'
    file_path_tree = 'models/decision_tree_model.pkl'
    # Abrir o arquivo .pkl e carregar o conteúdo
    try:
        model_random = joblib.load(file_path_random)
        model_tree = joblib.load(file_path_tree)
        print("Modelos carregados com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return

    # Verificar se o modelo tem o método 'predict'
    if not hasattr(model_random, 'predict'):
        print("O objeto carregado não é um modelo de sklearn.")
        return
        # Fazer previsões com o conjunto de teste e calcular a acurácia
    print("----------Random Forest Test-------------")
    rf_test_predictions = model_random.predict(X_test)
    rf_test_accuracy = accuracy_score(y_test, rf_test_predictions)
    rf_test_f1_score = f1_score(y_test, rf_test_predictions, average='weighted')
    print("Test Set Performance Random Forest:")
    print("Test Set Random Forest Accuracy: %s", rf_test_accuracy)
    print("Test Set Random Forest F1 Score: %s", rf_test_f1_score)

    # Verificar se o modelo tem o método 'predict'
    if not hasattr(model_tree, 'predict'):
        print("O objeto carregado não é um modelo de sklearn.")
        return
        # Fazer previsões com o conjunto de teste e calcular a acurácia
    print("----------Decision Tree Test-------------")
    rf_test_predictions = model_tree.predict(X_test)
    rf_test_accuracy = accuracy_score(y_test, rf_test_predictions)
    rf_test_f1_score = f1_score(y_test, rf_test_predictions, average='weighted')
    print("Test Set Performance Decision Tree:")
    print("Test Set Random Forest Accuracy: %s", rf_test_accuracy)
    print("Test Set Random Forest F1 Score: %s", rf_test_f1_score)
    print("-----------------------------------------------")
