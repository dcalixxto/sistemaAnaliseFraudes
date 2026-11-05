"""Configurações centrais do projeto (caminhos, colunas e rótulos).

Este módulo não executa nada: apenas define constantes que os outros arquivos
importam. Centralizar tudo aqui evita repetir caminhos e nomes em vários
lugares — mudar um valor aqui muda o comportamento do projeto inteiro.
"""
# pathlib.Path representa caminhos de arquivo de forma independente do sistema
# operacional (funciona igual no Windows, Linux e macOS).
from pathlib import Path

# ----------------------------------------------------------------------------
# Caminhos
# ----------------------------------------------------------------------------
# __file__ é o caminho deste arquivo (.../sistemaAnaliseFraudes/src/config.py).
# .resolve() transforma em caminho absoluto; o 1º .parent sobe para "src/" e o
# 2º .parent sobe para a raiz do projeto. Assim nenhum caminho fica "fixo" e o
# projeto roda em qualquer computador.
RAIZ = Path(__file__).resolve().parent.parent

# O operador "/" do pathlib junta pedaços de caminho: RAIZ / "data" = .../data
PASTA_DADOS = RAIZ / "data"            # onde ficam os CSVs originais do Kaggle
PASTA_ARTEFATOS = RAIZ / "artifacts"   # onde o treino grava modelos e métricas

ARQ_TREINO = PASTA_DADOS / "fraudTrain.csv"   # jan/2019 a jun/2020
ARQ_TESTE = PASTA_DADOS / "fraudTest.csv"     # jun/2020 a dez/2020
# Versão já processada da base, lida pela tela de análise exploratória.
ARQ_DADOS_EDA = PASTA_ARTEFATOS / "dados_processados.parquet"

# Semente aleatória: algoritmos que sorteiam (Random Forest, SMOTE, amostragens)
# produzem sempre o mesmo resultado quando a semente é fixa -> reprodutibilidade.
SEMENTE = 42

# ----------------------------------------------------------------------------
# Colunas
# ----------------------------------------------------------------------------
# Colunas brutas do Sparkov realmente usadas. Identificadores (cc_num, trans_num,
# nomes, rua, merchant) ficam de fora para evitar memorização/vazamento: o
# modelo deve aprender PADRÕES de fraude, não decorar cartões ou lojas.
COLUNAS_BRUTAS = [
    "trans_date_trans_time",  # data e hora da transação
    "category",               # categoria do comércio (14 valores)
    "amt",                    # valor em dólares
    "gender",                 # gênero do portador (F/M)
    "state",                  # estado do portador (só para o mapa do dashboard)
    "lat", "long",            # coordenadas do portador
    "city_pop",               # população da cidade do portador
    "dob",                    # data de nascimento (vira idade)
    "merch_lat", "merch_long",  # coordenadas do comerciante (viram distância)
    "is_fraud",               # alvo: 1 = fraude, 0 = legítima
]
ALVO = "is_fraud"  # nome da coluna que o modelo tenta prever

# Atributos que ENTRAM no modelo, criados em features.construir_atributos().
# Os numéricos serão padronizados (média 0, desvio 1)...
ATRIBUTOS_NUMERICOS = [
    "valor_log", "hora", "noturno", "dia_semana", "idade",
    "distancia_km", "city_pop_log", "lat", "long",
]
# ...e os categóricos (texto) serão convertidos em colunas 0/1 (one-hot).
ATRIBUTOS_CATEGORICOS = ["category", "gender"]

# ----------------------------------------------------------------------------
# Estratégias e modelos
# ----------------------------------------------------------------------------
# Dicionário código -> descrição. O código vira nome de pasta em artifacts/ e
# argumento de linha de comando; a descrição aparece no dashboard.
ESTRATEGIAS = {
    "pesos": "Pesos de classe (class_weight / scale_pos_weight)",
    "undersampling": "Subamostragem aleatória da classe majoritária (1:1)",
    "smote": "Superamostragem SMOTE (fraude = 10% da classe legítima)",
}

# Ordem em que os modelos aparecem em tabelas e gráficos.
MODELOS = ["Regressão Logística", "Naive Bayes", "Random Forest", "LightGBM"]

# ----------------------------------------------------------------------------
# Traduções (apenas visuais; o modelo continua usando os nomes originais)
# ----------------------------------------------------------------------------
CATEGORIAS_PT = {
    "entertainment": "Entretenimento",
    "food_dining": "Alimentação / Restaurantes",
    "gas_transport": "Combustível / Transporte",
    "grocery_net": "Supermercado (online)",
    "grocery_pos": "Supermercado (presencial)",
    "health_fitness": "Saúde / Fitness",
    "home": "Casa",
    "kids_pets": "Crianças / Pets",
    "misc_net": "Diversos (online)",
    "misc_pos": "Diversos (presencial)",
    "personal_care": "Cuidados pessoais",
    "shopping_net": "Compras (online)",
    "shopping_pos": "Compras (presencial)",
    "travel": "Viagem",
}

# O pandas numera os dias da semana de 0 (segunda) a 6 (domingo).
DIAS_SEMANA_PT = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
