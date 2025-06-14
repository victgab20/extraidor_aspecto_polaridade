import requests
import pandas as pd
import json
import os
import time
import gc

checkpoint_path = "aspect_polarity_gisela.csv"

# Verifica se já existe um checkpoint
if os.path.exists(checkpoint_path):
    data = pd.read_csv(checkpoint_path)
    print("Checkpoint carregado.")
else:
    data = pd.read_csv("train_data.csv")
    data["teste"] = ""

url = "http://localhost:11434/api/generate"
headers = {"Content-Type": "application/json"}

# Contador de progresso
save_interval = 100
contador_modificacoes = 0

for i in range(len(data)):
    if pd.notna(data.at[i, "teste"]) and data.at[i, "teste"] != "":
        continue  # Já processado

    print(f"Processando linha {i}...")
    
    prompt = f"""Identifique os aspectos explícitos mencionados no seguinte comentário e determine a polaridade (positivo, negativo ou neutro) de cada um, que são realmente importantes para entender se se trata de um comentário positivo, negativo ou neutro. Utilize apenas os aspectos que estão no texto do comentário, com limitação de apenas uma palavra, com exceção de locuções e palavras compostas; com sua respectiva polaridade, sem comentários adicionais. Apresente a saída no formato ['aspecto','polaridade']". gere apenas uma saída nesse modelo ['aspecto','polaridade']  Comentário: {data['texto'][i]}"""

    payload = {
        "model": "deepseek-r1:8b",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=30)
        response.raise_for_status()
        resposta = response.json()
        data.at[i, "teste"] = resposta.get("response", "")
        contador_modificacoes += 1
    except Exception as e:
        print(f"Erro na linha {i}: {e}")
        continue

    # Libera memória
    del response, resposta
    gc.collect()

    # Salva a cada 100 registros processados
    if contador_modificacoes >= save_interval:
        data.to_csv(checkpoint_path, index=False, header=True)
        print(f"Checkpoint salvo na linha {i}.")
        contador_modificacoes = 0

# Salva o restante ao final
data.to_csv(checkpoint_path, index=False, header=True)
print("Processamento concluído e arquivo salvo.")
