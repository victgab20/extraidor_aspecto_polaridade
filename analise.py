import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

df_humano = pd.read_csv(r"C:\Users\victo\Downloads\train2024 - train2024 (1).csv")
df_llm_gwen = pd.read_csv("gisela_llm1.csv")

print(df_humano)
print(df_llm_gwen)

df_humano.rename(columns={
    'texto': 'comentario',
    'aspect': 'aspecto',
    'polarity': 'polaridade'
}, inplace=True)

def limpar_texto(texto):
    return texto.lower().strip()

df_humano["comentario"] = df_humano["comentario"].apply(limpar_texto)
df_humano["aspecto"] = df_humano["aspecto"].apply(limpar_texto)
# a=lambda x: str(x).split(',')
# df_llm_gwen['aspecto_llm'].apply(a)
# print(df_llm_gwen["aspecto_llm"].describe())
# df_llm_gwen["aspecto_llm"] = df_llm_gwen["aspecto_llm"].apply(limpar_texto)

df_humano['chave'] = df_humano['comentario'] + ' || ' + df_humano['aspecto']
df_llm_gwen['chave'] = df_llm_gwen['comentario'] + ' || ' + df_llm_gwen['aspecto_llm']

df_comparado = df_humano.merge(
    df_llm_gwen[['chave', 'polaridade_llm']],
    on='chave',
    how='left'
)

df_comparado['igual_polaridade'] = df_comparado['polaridade'] == df_comparado['polaridade_llm']

total_comparados = df_comparado.shape[0]
iguais = df_comparado['igual_polaridade'].sum()
divergentes = total_comparados - iguais
acuracia = iguais / total_comparados


print(classification_report(
    df_comparado.dropna()['polaridade'], 
    df_comparado.dropna()['polaridade_llm']
))


divergentes = df_comparado[df_comparado['igual_polaridade'] == False]

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_true = df_comparado.dropna()['polaridade']
y_pred = df_comparado.dropna()['polaridade_llm']

cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negativo', 'Neutro', 'Positivo'])

disp.plot(cmap='Blues')
plt.title("Matriz de Confusão - Polaridade LLM vs Humano")
plt.show()


top_aspectos_llm = df_llm_gwen['aspecto_llm'].value_counts().head(10)

top_aspectos_llm.plot(kind='barh')
plt.xlabel('Frequência')
plt.ylabel('Aspecto')
plt.title('Top 10 Aspectos (LLM)')
plt.gca().invert_yaxis()
plt.show()



divergentes = df_comparado[df_comparado['igual_polaridade'] == False]
mais_divergentes = divergentes['aspecto'].value_counts().head(10)

mais_divergentes.plot(kind='bar')
plt.title('Aspectos com Mais Divergências (LLM vs Humano)')
plt.xlabel('Aspecto')
plt.ylabel('Quantidade de Divergências')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
