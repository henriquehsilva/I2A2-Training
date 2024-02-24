import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Carregar os dados do arquivo XLSX
df = pd.read_excel('datasets/Defective_Equipment (rev 2024-02-21).xlsx')
df.drop(columns=['Seq'], axis=1, inplace=True)
x = df.T.values

# Normalizando as variaveis preditoras
normalizer = MinMaxScaler(feature_range=(0, 1))
x_norm = normalizer.fit_transform(x)

# Transformando os dados para componentes PCA
pca = PCA(n_components=1)
x_pca = pca.fit_transform(x_norm)
y = [0] * len(x_pca)

# Variância explicada dos componentes
print(f"Variância explicada dos componentes: {pca.explained_variance_ratio_}")

# Plotar os dados
plt.figure(figsize=(18, 8))
plt.scatter(x_pca, y, label='Dados', s=200)

# Plotar a reta horizontal centralizada
plt.axhline(0, color='red', linestyle='--', label='Média da Variância Explicada')

# Adicionar rótulos
labels = df.columns
for i, label in enumerate(labels):
    plt.text(x_pca[i], y[i], ' - '+label, rotation=45)

# Configurar labels e título
plt.title('Dados e Média da Variância Explicada do Componente Principal')
plt.gca().get_xaxis().set_visible(False)
plt.gca().get_yaxis().set_visible(False)
plt.legend()

# Exibir o gráfico
plt.show()
