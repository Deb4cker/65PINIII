import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Carregar dados
fruits_file_path = 'data/frutas.csv'
fruits_data = pd.read_csv(fruits_file_path)

# Separar características (X) e classes (y)
X = fruits_data.drop(columns=['classe'])
y = fruits_data['classe']

# Dividir os dados em treinamento (70%) e teste + validação (30%)
X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Dividir dados de teste + validação em teste (15%) e validação (15%)
X_test, X_validation, y_test, y_validation = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42, stratify=y_test_val)

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

# Inicializar o modelo GradientBoostingClassifier
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)

# Fazer previsões com o conjunto de validação
gb_val_predictions = gb_model.predict(X_validation)
gb_val_accuracy = accuracy_score(y_validation, gb_val_predictions)
print("Acurácia do modelo GradientBoosting no conjunto de validação:", gb_val_accuracy)

# Fazer previsões com o conjunto de validação
gb_test_predictions = gb_model.predict(X_test)
gb_test_accuracy = accuracy_score(y_test, gb_test_predictions)
print("Acurácia do modelo GradientBoosting no conjunto de test:", gb_test_accuracy)

# Inicializar o modelo SVC
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Fazer previsões com o conjunto de validação
svm_val_predictions = svm_model.predict(X_validation)
svm_val_accuracy = accuracy_score(y_validation, svm_val_predictions)
print("Acurácia do modelo SVM no conjunto de validação:", svm_val_accuracy)

# Fazer previsões com o conjunto de teste
svm_test_predictions = svm_model.predict(X_test)
svm_test_accuracy = accuracy_score(y_validation, svm_val_predictions)
print("Acurácia do modelo SVM no conjunto de teste:", svm_test_accuracy)

# Calcular o F1-score para as previsões do modelo RandomForest
rf_f1_score = f1_score(y_validation, rf_val_predictions, average='weighted')
print("F1-score do modelo RandomForest no conjunto de validação:", rf_f1_score)

# Calcular o F1-score para as previsões do modelo DecisionTree
dt_f1_score = f1_score(y_validation, dt_val_predictions, average='weighted')
print("F1-score do modelo DecisionTree no conjunto de validação:", dt_f1_score)

# Calcular o F1-score para as previsões do modelo GradientBoosting
gb_f1_score = f1_score(y_validation, gb_val_predictions, average='weighted')
print("F1-score do modelo GradientBoosting no conjunto de validação:", gb_f1_score)

# Calcular o F1-score para as previsões do modelo SVM
svm_f1_score = f1_score(y_validation, svm_val_predictions, average='weighted')
print("F1-score do modelo SVM no conjunto de validação:", svm_f1_score)

# Salvar o modelo de machine learning na pasta /models

joblib.dump(fruits_random_forest_model, 'models/random_forest_model.pkl')
joblib.dump(fruits_decision_tree_model, 'models/decision_tree_model.pkl')