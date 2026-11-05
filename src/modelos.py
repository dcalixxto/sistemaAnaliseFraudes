"""Definição do pré-processamento, dos 4 classificadores e do balanceamento.

Este módulo diz O QUE será treinado; quem executa o treino é src/treinar.py.
"""
import numpy as np
# imbalanced-learn: técnicas de balanceamento compatíveis com o scikit-learn.
from imblearn.over_sampling import SMOTE                 # cria fraudes sintéticas
from imblearn.under_sampling import RandomUnderSampler   # descarta legítimas
from lightgbm import LGBMClassifier                      # Gradient Boosting (Microsoft)
from sklearn.compose import ColumnTransformer            # transformação por grupo de colunas
from sklearn.ensemble import RandomForestClassifier      # Bagging de árvores
from sklearn.linear_model import LogisticRegression      # modelo linear probabilístico
from sklearn.naive_bayes import GaussianNB               # Teorema de Bayes + normal
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ATRIBUTOS_CATEGORICOS, ATRIBUTOS_NUMERICOS, SEMENTE


def criar_preprocessador(numericos=ATRIBUTOS_NUMERICOS, categoricos=ATRIBUTOS_CATEGORICOS):
    """Padronização dos contínuos + one-hot das categóricas (idêntico para os 4).

    O ColumnTransformer aplica uma transformação diferente a cada grupo de
    colunas e junta o resultado numa única matriz (9 numéricas + 14 categorias
    + 2 gêneros = 25 colunas).
    """
    return ColumnTransformer([
        # (nome, transformador, colunas)
        # StandardScaler: z = (x - média) / desvio  ->  média 0 e desvio 1.
        # Essencial para a Regressão Logística; indiferente para árvores.
        ("num", StandardScaler(), numericos),
        # OneHotEncoder: "travel" vira category_travel=1 e as demais=0.
        #   handle_unknown="ignore": categoria nunca vista gera tudo 0, sem erro.
        #   sparse_output=False: matriz comum (densa); o GaussianNB não aceita esparsa.
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricos),
    ])


def criar_modelos(usar_pesos, razao_neg_pos):
    """Os 4 algoritmos, um de cada família.

    Com `usar_pesos=True` a classe minoritária é penalizada pelo próprio
    algoritmo; caso contrário o balanceamento já foi feito por reamostragem.
    O Naive Bayes não tem `class_weight`: recebe `sample_weight` no fit.

    razao_neg_pos = nº de legítimas / nº de fraudes (usado pelo LightGBM).
    """
    # "balanced" = peso de cada classe inversamente proporcional à sua frequência.
    peso = "balanced" if usar_pesos else None
    return {
        # Linear: soma ponderada dos atributos -> sigmoide -> probabilidade 0..1.
        # max_iter=1000 dá mais iterações para o otimizador convergir.
        "Regressão Logística": LogisticRegression(
            class_weight=peso, max_iter=1000, random_state=SEMENTE),
        # Bayes "ingênuo": supõe atributos independentes e com distribuição normal.
        "Naive Bayes": GaussianNB(),
        # Bagging: várias árvores treinadas em amostras diferentes votam.
        "Random Forest": RandomForestClassifier(
            n_estimators=150,        # nº de árvores
            max_depth=20,            # profundidade máxima (limita sobreajuste)
            min_samples_leaf=5,      # cada folha precisa de pelo menos 5 exemplos
            max_samples=0.3,         # cada árvore vê 30% das linhas (acelera)
            # balanced_subsample: pesos recalculados na amostra de cada árvore.
            class_weight="balanced_subsample" if usar_pesos else None,
            n_jobs=-1,               # usa todos os núcleos do processador
            random_state=SEMENTE),
        # Boosting: árvores em sequência, cada uma corrige o erro das anteriores.
        "LightGBM": LGBMClassifier(
            n_estimators=500,        # nº de árvores (rodadas de boosting)
            learning_rate=0.05,      # quanto cada árvore corrige (menor = mais estável)
            num_leaves=63,           # tamanho (nº de folhas) de cada árvore
            subsample=0.8, subsample_freq=1,  # cada árvore vê 80% das linhas
            colsample_bytree=0.8,    # ...e 80% das colunas
            # Quantas vezes uma fraude pesa mais que uma legítima (~172 na base real).
            scale_pos_weight=razao_neg_pos if usar_pesos else 1.0,
            n_jobs=-1, random_state=SEMENTE,
            verbose=-1),             # silencia as mensagens do LightGBM
    }


def reamostrar(X, y, estrategia):
    """Aplica o balanceamento SOMENTE no conjunto de treino já pré-processado.

    O teste nunca passa por aqui: ele precisa manter a proporção real de fraudes
    para que as métricas representem o mundo real.
    """
    if estrategia == "undersampling":
        # sampling_strategy=1.0 -> sorteia legítimas até ficarem 1:1 com as fraudes.
        return RandomUnderSampler(sampling_strategy=1.0, random_state=SEMENTE).fit_resample(X, y)
    if estrategia == "smote":
        # sampling_strategy=0.1 -> cria fraudes sintéticas até elas somarem 10%
        # das legítimas. Cada nova fraude é um ponto entre uma fraude real e uma
        # de suas vizinhas mais próximas (k-vizinhos).
        return SMOTE(sampling_strategy=0.1, random_state=SEMENTE).fit_resample(X, y)
    # "pesos": dados inalterados; o balanceamento fica a cargo dos modelos.
    return X, y


def pesos_amostra(y):
    """Peso inverso à frequência da classe (equivalente a class_weight='balanced').

    peso_classe = n_total / (n_classes * n_da_classe)
    Na base real: legítima ≈ 0,50 e fraude ≈ 86,4 — as fraudes, somadas, passam
    a pesar o mesmo que as legítimas.
    """
    y = np.asarray(y)
    freq = np.bincount(y)                        # [nº de zeros, nº de uns]
    # Calcula o peso de cada classe e "espalha" para cada linha conforme seu y.
    return (len(y) / (len(freq) * freq))[y]
