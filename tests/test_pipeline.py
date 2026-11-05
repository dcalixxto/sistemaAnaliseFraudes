"""Testes das partes do pipeline que sustentam a metodologia. Rode: python -m pytest"""
import numpy as np
import pandas as pd
import pytest

from src.config import ATRIBUTOS_CATEGORICOS, ATRIBUTOS_NUMERICOS
from src.features import construir_atributos, haversine_km, separar_xy
from src.modelos import criar_preprocessador, pesos_amostra, reamostrar


def bruto():
    n = 4
    return pd.DataFrame({
        "trans_date_trans_time": ["2020-01-01 23:30:00", "2020-01-02 04:00:00",
                                  "2020-01-03 12:00:00", "2020-01-04 03:59:00"],
        "category": ["travel", "home", "travel", "home"],
        "amt": [10.0, 100.0, 1000.0, 0.0],
        "gender": ["F", "M", "F", "M"],
        "lat": [40.0] * n, "long": [-75.0] * n, "city_pop": [1000] * n,
        "dob": ["1990-01-01"] * n,
        "merch_lat": [41.0] * n, "merch_long": [-75.0] * n,
    })


def test_haversine_zero_e_um_grau_de_latitude():
    assert haversine_km(10, 20, 10, 20) == pytest.approx(0)
    assert haversine_km(40, -75, 41, -75) == pytest.approx(111.19, abs=0.01)


def test_atributos_temporais_e_janela_noturna():
    df = construir_atributos(bruto())
    assert df["hora"].tolist() == [23, 4, 12, 3]
    assert df["noturno"].tolist() == [1, 0, 0, 1]   # 22h a 3h, inclusive
    assert df["dia_semana"].iloc[0] == 2            # 01/01/2020 foi uma quarta-feira
    assert df["idade"].iloc[0] == pytest.approx(30, abs=0.01)
    assert df["distancia_km"].iloc[0] == pytest.approx(111.19, abs=0.01)


def test_nao_altera_a_tabela_original():
    original = bruto()
    colunas = list(original.columns)
    construir_atributos(original)
    assert list(original.columns) == colunas


def test_preprocessador_usa_so_estatisticas_do_treino():
    n = 200
    rng = np.random.default_rng(0)
    treino = pd.DataFrame({c: rng.normal(size=n) for c in ATRIBUTOS_NUMERICOS})
    treino["category"] = (["a", "b"] * n)[:n]
    treino["gender"] = (["F", "M"] * n)[:n]
    prep = criar_preprocessador().fit(treino)
    medias = prep.named_transformers_["num"].mean_.copy()
    teste = treino.head(5).copy()
    teste[ATRIBUTOS_NUMERICOS] = 1000.0
    teste["category"] = "nunca_vista"
    saida = prep.transform(teste)             # categoria nova não pode gerar erro
    assert np.array_equal(prep.named_transformers_["num"].mean_, medias)
    assert saida.shape == (5, len(ATRIBUTOS_NUMERICOS) + 2 + 2)


def test_subamostragem_iguala_as_classes():
    rng = np.random.default_rng(0)
    y = np.array([0] * 900 + [1] * 100)
    _, yu = reamostrar(rng.normal(size=(1000, 3)), y, "undersampling")
    assert (yu == 0).sum() == (yu == 1).sum() == 100


def test_smote_leva_fraudes_a_10_por_cento_das_legitimas():
    rng = np.random.default_rng(0)
    y = np.array([0] * 1000 + [1] * 20)
    _, ys = reamostrar(rng.normal(size=(1020, 3)), y, "smote")
    assert (ys == 0).sum() == 1000 and (ys == 1).sum() == 100


def test_estrategia_de_pesos_nao_altera_os_dados():
    X, y = np.zeros((4, 2)), np.array([0, 0, 0, 1])
    Xp, yp = reamostrar(X, y, "pesos")
    assert Xp is X and yp is y


def test_pesos_igualam_as_classes():
    y = np.array([0] * 90 + [1] * 10)
    p = pesos_amostra(y)
    assert p[y == 0].sum() == pytest.approx(p[y == 1].sum())


def test_separar_xy_sem_rotulo():
    X, y = separar_xy(construir_atributos(bruto()))
    assert y is None
    assert list(X.columns) == ATRIBUTOS_NUMERICOS + ATRIBUTOS_CATEGORICOS
