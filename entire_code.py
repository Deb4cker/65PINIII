import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import DecisionTree
import RandomForest
# Carregar dado
fruits_file_path = 'data/frutas.csv'
fruits_data = pd.read_csv(fruits_file_path)

# Separar características (X) e classes (y)
X = fruits_data.drop(columns=['classe'])
y = fruits_data['classe']

X_resampled, y_resampled = SMOTE().fit_resample(X, y)

# Dividir os dados em treinamento (70%) e teste + validação (30%)
X_train, X_test_val, y_train, y_test_val = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)

# Dividir dados de teste + validação em teste (15%) e validação (15%)
X_test, X_validation, y_test, y_validation = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42, stratify=y_test_val)

tree = DecisionTree
tree.initTrain(X_train, y_train, X_validation, y_validation, X_test, y_test)

forest = RandomForest
forest.initTrain(X_train, y_train, X_validation, y_validation, X_test, y_test)
