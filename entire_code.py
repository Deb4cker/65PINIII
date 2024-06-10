import pandas as pd
from sklearn.model_selection import train_test_split
import DecisionTree
import RandomForest
import matplotlib.pyplot as plt
import TestModels

# importar as bibliotecas necessárias
from imblearn.under_sampling import RandomUnderSampler

# Carregar dado
fruits_file_path = 'data/frutas.csv'
fruits_data = pd.read_csv(fruits_file_path)

# Separar características (X) e classes (y)
X = fruits_data.drop(columns=['classe'])
y = fruits_data['classe']


# criando uma instância do RandomUnderSampling
rus = RandomUnderSampler(random_state=42, sampling_strategy='not minority')

X_resampled, y_resampled = rus.fit_resample(X, y)

# Plotar os gráficos antes e depois do balanceamento
fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
autopct = "%.2f%%"

# Plotar dados originais
fruits_data['classe'].value_counts().plot.pie(autopct=autopct, ax=axs[0], colors=['skyblue', 'orange', 'lightgreen', 'lightcoral'], wedgeprops=dict(edgecolor='black'))
axs[0].set_title("Original Data")
axs[0].set_ylabel("")  # Remover label padrão do eixo y

# Plotar dados balanceados
y_resampled.value_counts().plot.pie(autopct=autopct, ax=axs[1], colors=['skyblue', 'orange', 'lightgreen', 'lightcoral'], wedgeprops=dict(edgecolor='black'))
axs[1].set_title("Balanced Data")
axs[1].set_ylabel("")  # Remover label padrão do eixo y

fig.tight_layout()

# Salvar a figura em um arquivo
fig.savefig('Images/output.png')

# Opcional: exibir uma mensagem para confirmar que a figura foi salva
print("Figura salva como output.png")

# Dividir os dados em treinamento (70%) e teste + validação (30%)
X_train, X_test_val, y_train, y_test_val = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)

# Dividir dados de teste + validação em teste (15%) e validação (15%)
X_test, X_validation, y_test, y_validation = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42, stratify=y_test_val)

tree = DecisionTree
tree.initTrain(X_train, y_train, X_validation, y_validation)

forest = RandomForest
forest.initTrain(X_train, y_train, X_validation, y_validation)

test = TestModels
test.init(X_test, y_test)

