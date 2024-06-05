import numpy as np
import pandas as pd

# Ler o arquivo CSV no DataFrame
data = pd.read_csv('data/frutas.csv')

# Obter o número de amostras
num_samples = len(data)

# Gerar novos dados aleatórios para cada coluna


#['area','perimetro','eixo_maior','eixo_menor','excentricidade','eqdiasq','solidez','area_convexa',
#                     'extensao','proporcao','redondidade','compactidade','fator_forma_1','fator_forma_2','fator_forma_3',
#                     'fator_forma_4','RR_media','RG_media','RB_media','RR_dev','RG_dev','RB_dev','RR_inclinacao','RG_inclinacao',
#                     'RB_inclinacao','RR_curtose','RG_curtose','RB_curtose','RR_entropia','RG_entropia','RB_entropia','RR_all','RG_all','RB_all']
new_data = pd.DataFrame({
    'area': np.random.randint(low=data['area'].min(), high=data['area'].max(), size=num_samples),
    'perimetro': np.random.uniform(low=data['perimetro'].min(), high=data['perimetro'].max(), size=num_samples),
    'eixo_maior': np.random.uniform(low=data['eixo_maior'].min(), high=data['eixo_maior'].max(), size=num_samples),
    'eixo_menor': np.random.uniform(low=data['eixo_menor'].min(), high=data['eixo_menor'].max(), size=num_samples),
    'excentricidade': np.random.uniform(low=data['excentricidade'].min(), high=data['excentricidade'].max(), size=num_samples),
    'eqdiasq': np.random.uniform(low=data['eqdiasq'].min(), high=data['eqdiasq'].max(), size=num_samples),
    'solidez': np.random.uniform(low=data['solidez'].min(), high=data['solidez'].max(), size=num_samples),
    'area_convexa': np.random.uniform(low=data['area_convexa'].min(), high=data['area_convexa'].max(), size=num_samples),
    'extensao': np.random.uniform(low=data['extensao'].min(), high=data['extensao'].max(), size=num_samples),
    'proporcao': np.random.uniform(low=data['proporcao'].min(), high=data['proporcao'].max(), size=num_samples),
    'redondidade': np.random.uniform(low=data['redondidade'].min(), high=data['redondidade'].max(), size=num_samples),
    'compactidade': np.random.uniform(low=data['compactidade'].min(), high=data['compactidade'].max(), size=num_samples),
    'fator_forma_1': np.random.uniform(low=data['fator_forma_1'].min(), high=data['fator_forma_1'].max(), size=num_samples),
    'fator_forma_2': np.random.uniform(low=data['fator_forma_2'].min(), high=data['fator_forma_2'].max(), size=num_samples),
    'fator_forma_3': np.random.uniform(low=data['fator_forma_3'].min(), high=data['fator_forma_3'].max(), size=num_samples),
    'fator_forma_4': np.random.uniform(low=data['fator_forma_4'].min(), high=data['fator_forma_4'].max(), size=num_samples),
    'RR_media': np.random.uniform(low=data['RR_media'].min(), high=data['RR_media'].max(), size=num_samples),
    'RG_media': np.random.uniform(low=data['RG_media'].min(), high=data['RG_media'].max(), size=num_samples),
    'RB_media': np.random.uniform(low=data['RB_media'].min(), high=data['RB_media'].max(), size=num_samples),
    'RR_dev': np.random.uniform(low=data['RR_dev'].min(), high=data['RR_dev'].max(), size=num_samples),
    'RG_dev': np.random.uniform(low=data['RG_dev'].min(), high=data['RG_dev'].max(), size=num_samples),
    'RB_dev': np.random.uniform(low=data['RB_dev'].min(), high=data['RB_dev'].max(), size=num_samples),
    'RR_inclinacao': np.random.uniform(low=data['RR_inclinacao'].min(), high=data['RR_inclinacao'].max(), size=num_samples),
    'RG_inclinacao': np.random.uniform(low=data['RG_inclinacao'].min(), high=data['RG_inclinacao'].max(), size=num_samples),
    'RB_inclinacao': np.random.uniform(low=data['RB_inclinacao'].min(), high=data['RB_inclinacao'].max(), size=num_samples),
    'RR_curtose': np.random.uniform(low=data['RR_curtose'].min(), high=data['RR_curtose'].max(), size=num_samples),
    'RG_curtose': np.random.uniform(low=data['RG_curtose'].min(), high=data['RG_curtose'].max(), size=num_samples),
    'RB_curtose': np.random.uniform(low=data['RB_curtose'].min(), high=data['RB_curtose'].max(), size=num_samples),
    'RR_entropia': np.random.uniform(low=data['RR_entropia'].min(), high=data['RR_entropia'].max(), size=num_samples),
    'RG_entropia': np.random.uniform(low=data['RG_entropia'].min(), high=data['RG_entropia'].max(), size=num_samples),
    'RB_entropia': np.random.uniform(low=data['RB_entropia'].min(), high=data['RB_entropia'].max(), size=num_samples),
    'RR_all': np.random.uniform(low=data['RR_all'].min(), high=data['RR_all'].max(), size=num_samples),
    'RG_all': np.random.uniform(low=data['RG_all'].min(), high=data['RG_all'].max(), size=num_samples),
    'RB_all': np.random.uniform(low=data['RB_all'].min(), high=data['RB_all'].max(), size=num_samples)
})

# Salvar os novos dados em um arquivo CSV
new_data.to_csv('teste/TestCSV.csv', index=False)
