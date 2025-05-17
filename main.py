import requests
import pandas as pd
import json


data = pd.read_csv(r"train_data.csv")

url = "http://localhost:11434/api/generate"
headers = {
    "Content-Type": "application/json"
}
print(data)
for i in range(len(data)):
    print(i)
    payload = {
        "model": "qwen2.5",
        "prompt": f"""Identifique os aspectos explícitos mencionados no seguinte comentário e determine a polaridade (positivo, negativo ou neutro) de cada um, que são realmente importantes para entender se se trata de um comentário positivo, negativo ou neutro. Utilize apenas os aspectos que estão no texto do comentário, com limitação de apenas uma palavra, com exceção de locuções e palavras compostas; com sua respectiva polaridade, sem comentários adicionais. Apresente a saída no formato ['aspecto','polaridade']". gere apenas uma saída nesse modelo ['aspecto','polaridade']  Comentário: {data['texto'][i]}""",
        "stream": False
    }

    response = requests.post(url, data=json.dumps(payload), headers=headers)
    resposta = response.json()

    data.at[i, "teste"] = resposta["response"]
print(data)
data.to_csv('aspect_polarity_gisela.csv', index=False, header=True)
