import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

target_text = open('datasets/Base Text - Matrix Representation I - v2.txt', 'r').read()
compared_text = open('datasets/Base Text - Matrix Representation II - v2.txt', 'r').read()

# Número de n-gramas
n = 1

# Instância o contador de n-gramas
counts = CountVectorizer(analyzer='word', ngram_range=(n,n))

# Cria um dicionário de n-gramas
vocab2int = counts.fit([compared_text, target_text]).vocabulary_

# Cria uma matriz de contagem de n-gramas para dois textos
n_gramas = counts.fit_transform([compared_text, target_text])

n_gramas_array = n_gramas.toarray()

print(f"Vetor de n-gramas:\n {n_gramas_array}\n")
print(f"Dicionario de n-gramas (bigrama):\n {vocab2int}\n")

intersection_list = np.amin(n_gramas_array, axis=0)
intersection_count = np.sum(intersection_list)

index_A = 0
A_count = np.sum(n_gramas_array[index_A])

print(f"O grau de similaridade entre os textos é: {(intersection_count/A_count):.2f}")