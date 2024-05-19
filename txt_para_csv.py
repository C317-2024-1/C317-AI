import os
import pandas as pd

dados = []

for arquivo in os.listdir('textos'):
    if arquivo.endswith(".txt"):
        nome_arquivo = os.path.splitext(arquivo)[0]
        caminho_arquivo = os.path.join('textos', arquivo)
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            conteudo = f.read()
            conteudo_tratado = conteudo[21:-30]
            dados.append([nome_arquivo, conteudo_tratado])

df = pd.DataFrame(dados, columns=['title', 'text'])
df.to_csv('dados/titulo_e_texto.csv', index=False, encoding='utf-8')
