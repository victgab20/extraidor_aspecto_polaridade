import pandas as pd

import ast
import re

data = pd.read_csv(r"C:\Users\victo\Downloads\aspect_polarity_gisela.csv")
# def pre_process(dataframe: pd.DataFrame):
#     dataframe["label"] = dataframe["label"].apply(lambda row: 0 if row == "fake" else 1)

#     return dataframe
# d = pre_process(data)
def extrair_primeira_lista(texto):
    listas = re.findall(r"\[.*?\]", texto)
    if listas:
        return ast.literal_eval(listas[0])
    return None

# data['teste'] = data['teste'].apply(extrair_primeira_lista)
# data[['aspecto_llm', 'polaridade_llm']] = pd.DataFrame(
#     data['teste'].apply(lambda x: x if isinstance(x, list) and len(x) == 2 else [None, None]).tolist(),
#     index=data.index
# )
# data.drop('teste',axis=1, inplace=True)
# data['polaridade_llm'] = data['polaridade_llm'].apply(lambda x: 1 if x == 'positivo' else -1)
# print(data)


# data.to_csv('gisela_tratado.csv',index=False)

novos_dados = []

# Itera sobre as linhas do DataFrame original
for _, row in data.iterrows():
    comentario = row['texto']
    
    # Divide os múltiplos pares de aspecto/polaridade
    linhas_teste = row['teste'].split("\n")
    
    for linha in linhas_teste:
        try:
            aspecto_polaridade = ast.literal_eval(linha)  # Converte string para lista
            novos_dados.append({'comentario': comentario, 'aspecto_polaridade': aspecto_polaridade})
        except:
            continue

# Cria novo DataFrame com os dados processados
df_final = pd.DataFrame(novos_dados)

# Exibe os primeiros resultados
# print(df_final.head())

# print(data["teste"][0])

print(df_final)


#df_final['aspecto_polaridade'] = df_final['aspecto_polaridade'].apply(extrair_primeira_lista)
df_final[['aspecto_llm', 'polaridade_llm']] = pd.DataFrame(
    df_final['aspecto_polaridade'].apply(lambda x: x if isinstance(x, list) and len(x) == 2 else [None, None]).tolist(),
    index=df_final.index
)
# data.drop('aspecto_polaridade',axis=1, inplace=True)
df_final.drop('aspecto_polaridade', axis=1,inplace=True)
df_final['polaridade_llm'] = df_final['polaridade_llm'].apply(lambda x: 1 if x == 'positivo' else -1)
print(df_final)

df_final.to_csv('gisela_llm1.csv')