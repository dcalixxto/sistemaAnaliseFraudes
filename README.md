# Sistema Web para Análise de Fraudes em Cartões de Crédito

TCC de Daniel Calixto Lima (UNIPAC). Reúne, em um painel Streamlit, a análise
exploratória da base sintética Sparkov (1.852.394 transações, 0,52% de fraudes) e a
comparação de quatro algoritmos de classificação sob três estratégias de tratamento
do desbalanceamento. O sistema analisa dados históricos e não classifica transações
individuais.

## Como executar

Requer Python 3.13.

```bash
pip install -r requirements.txt
```

1. Baixe o [dataset no Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
   e coloque `fraudTrain.csv` e `fraudTest.csv` em `data/`.
2. Treine e avalie as 12 combinações (gera `artifacts/`):

```bash
python -m src.treinar --estrategia todas
```

3. Gere as análises complementares citadas no artigo (opcional):

```bash
python -m src.analises
```

4. Abra o painel:

```bash
streamlit run app.py
```

Para rodar os testes, instale o pytest (`pip install pytest`) e execute:

```bash
python -m pytest
```

## Estrutura

| Arquivo | Papel |
|---|---|
| `src/config.py` | Caminhos, colunas usadas e rótulos em português |
| `src/features.py` | Carga dos CSVs e engenharia de atributos |
| `src/modelos.py` | Pré-processamento, os 4 classificadores e o balanceamento |
| `src/treinar.py` | Treino, avaliação no teste, tempos e gravação dos artefatos |
| `src/analises.py` | Qualidade dos dados, sensibilidade aos atributos e incerteza |
| `app.py` | Painel Streamlit |
| `tests/` | Testes do pipeline |

## Decisões metodológicas

- **Divisão cronológica.** Treino = `fraudTrain.csv` (jan/2019 a jun/2020) e teste =
  `fraudTest.csv` (jun a dez/2020).
- **Atributos.** Valor (log), hora, indicador noturno (22h a 3h), dia da semana, idade do
  portador, distância portador–comerciante (Haversine), população da cidade (log),
  latitude e longitude do portador, categoria e gênero. Número do cartão, nome, endereço e
  nome do comércio são descartados.
- **Pré-processamento.** `StandardScaler` e `OneHotEncoder`, ajustados só no treino.
- **Balanceamento, só no treino.** Pesos por classe, subamostragem 1:1 e SMOTE até as
  fraudes valerem 10% das transações legítimas.
- **Métricas.** PR-AUC (principal), Recall, Precisão, F1 e AUC-ROC (complementar). A acurácia
  não é usada. O controle de limiar do painel é exploratório: escolher o limiar olhando o
  teste seria vazamento.

## Limitações

Os dados são sintéticos, e os valores obtidos não se transferem diretamente para fraudes
reais. Os hiperparâmetros foram fixados sem busca sistemática. O painel ainda não foi
avaliado com usuários.
