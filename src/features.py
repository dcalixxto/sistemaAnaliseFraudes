"""Carga dos dados brutos e engenharia de atributos.

Transforma as colunas brutas do Sparkov nos 11 atributos usados pelos modelos.
"""
import numpy as np    # cálculo vetorizado (opera em milhões de linhas de uma vez)
import pandas as pd   # tabelas (DataFrames)

from src.config import (
    ALVO, ARQ_TESTE, ARQ_TREINO, ATRIBUTOS_CATEGORICOS, ATRIBUTOS_NUMERICOS,
    COLUNAS_BRUTAS,
)


def haversine_km(lat1, lon1, lat2, lon2):
    """Distância em km sobre a superfície terrestre entre dois pontos (graus).

    Fórmula de Haversine: d = 2R * arcsin( sqrt( sin²(Δφ/2) + cos φ1 cos φ2 sin²(Δλ/2) ) )
    onde φ = latitude, λ = longitude (em radianos) e R = 6371 km (raio médio da Terra).
    Aceita números isolados ou colunas inteiras (Series/arrays).
    """
    # Seno e cosseno do NumPy trabalham em radianos; converte os 4 valores de graus.
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    # Termo "a" da fórmula (a parte dentro da raiz quadrada).
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    # Ângulo central * raio da Terra = distância em km.
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def carregar_bruto():
    """Lê fraudTrain.csv/fraudTest.csv do Kaggle (Sparkov).

    Devolve um único DataFrame com a coluna extra "conjunto" ("treino"/"teste").
    Se só existir o arquivo de treino, faz um corte temporal 70/30 para que o
    teste continue sendo "o futuro" em relação ao treino.
    """
    # Sem o arquivo de treino não há o que fazer: erro com instrução clara.
    if not ARQ_TREINO.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {ARQ_TREINO}\n"
            "Baixe o dataset em https://www.kaggle.com/datasets/kartik2112/fraud-detection "
            "e coloque fraudTrain.csv e fraudTest.csv na pasta data/."
        )
    # usecols = lê só as 12 colunas úteis (economiza memória e tempo).
    treino = pd.read_csv(ARQ_TREINO, usecols=COLUNAS_BRUTAS)
    if ARQ_TESTE.exists():
        teste = pd.read_csv(ARQ_TESTE, usecols=COLUNAS_BRUTAS)
        # Marca a origem de cada linha antes de juntar as duas tabelas.
        treino["conjunto"], teste["conjunto"] = "treino", "teste"
        # Empilha uma tabela embaixo da outra; ignore_index renumera as linhas.
        df = pd.concat([treino, teste], ignore_index=True)
    else:
        # Plano B: ordena por data e usa os primeiros 70% como treino.
        # (Comparar strings "AAAA-MM-DD HH:MM:SS" já ordena cronologicamente.)
        df = treino.sort_values("trans_date_trans_time", ignore_index=True)
        corte = int(len(df) * 0.7)
        # np.where(condição, valor_se_verdadeiro, valor_se_falso), linha a linha.
        df["conjunto"] = np.where(df.index < corte, "treino", "teste")
    # Converte texto -> tipo data, o que permite extrair hora, dia, fazer contas.
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])
    return df


def construir_atributos(df):
    """Recebe colunas brutas do Sparkov e devolve o DataFrame com os atributos."""
    df = df.copy()  # não altera a tabela original de quem chamou a função
    dt = pd.to_datetime(df["trans_date_trans_time"])  # garante o tipo data

    # Hora do dia (0-23): fraudes se concentram entre 22h e 3h.
    df["hora"] = dt.dt.hour
    # Indicador 0/1 de horário noturno. Um modelo linear não "percebe" que 23h e
    # 1h são próximas (23 e 1 são números distantes); este sinal resolve isso.
    df["noturno"] = df["hora"].isin([22, 23, 0, 1, 2, 3]).astype(int)
    # Dia da semana: 0 = segunda ... 6 = domingo.
    df["dia_semana"] = dt.dt.dayofweek
    # Idade em anos na data da compra (365,25 compensa os anos bissextos).
    df["idade"] = (dt - pd.to_datetime(df["dob"])).dt.days / 365.25
    # Distância portador -> comerciante em km (hipótese: fraude acontece longe).
    df["distancia_km"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
    # log(1 + x) comprime valores muito grandes; reduz a assimetria do valor
    # (maioria < US$ 100, alguns > US$ 20.000) e da população (dezenas a milhões).
    df["valor_log"] = np.log1p(df["amt"])
    df["city_pop_log"] = np.log1p(df["city_pop"])
    return df


def separar_xy(df):
    """Separa a tabela em X (atributos de entrada) e y (alvo 0/1)."""
    # A ordem das colunas segue config.py e precisa ser igual no treino e no uso.
    X = df[ATRIBUTOS_NUMERICOS + ATRIBUTOS_CATEGORICOS]
    # Se não houver coluna is_fraud (dados novos, sem rótulo), y = None.
    y = df[ALVO].astype(int) if ALVO in df else None
    return X, y
