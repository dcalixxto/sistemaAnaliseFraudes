"""Análises complementares citadas no artigo. Rode depois de `python -m src.treinar --estrategia todas`.

Uso:
    python -m src.analises

Calcula e grava em artifacts/analises_complementares.json:
  1. qualidade dos dados (nulos, duplicatas, cartões em comum entre treino e teste);
  2. taxa de fraude por hora só no treino (confirma a janela noturna sem olhar o teste);
  3. sensibilidade do LightGBM com SMOTE à retirada de atributos do portador;
  4. intervalo de confiança da PR-AUC por bootstrap no conjunto de teste.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import (
    ALVO, ARQ_TESTE, ARQ_TREINO, ATRIBUTOS_CATEGORICOS, ATRIBUTOS_NUMERICOS,
    PASTA_ARTEFATOS, SEMENTE,
)
from src.features import carregar_bruto, construir_atributos
from src.modelos import criar_modelos, criar_preprocessador, reamostrar

N_BOOTSTRAP = 200

# Conjuntos de atributos retirados no teste de sensibilidade: (numéricos, categóricos).
LOCALIZACAO = ["lat", "long", "city_pop_log"]
CENARIOS = {
    "completo": ([], []),
    "sem localização": (LOCALIZACAO, []),
    "sem localização e idade": (LOCALIZACAO + ["idade"], []),
    "sem localização, idade e gênero": (LOCALIZACAO + ["idade"], ["gender"]),
}


def qualidade_dos_dados():
    """Lê as colunas de identificação, que o treino descarta, só para esta verificação."""
    tr = pd.read_csv(ARQ_TREINO, index_col=0)
    te = pd.read_csv(ARQ_TESTE, index_col=0)
    todos = pd.concat([tr, te], ignore_index=True)
    so_no_teste = ~te["cc_num"].isin(set(tr["cc_num"]))
    exclusivos = te[so_no_teste]
    return {
        "valores_nulos": int(todos.isna().sum().sum()),
        "linhas_duplicadas": int(todos.duplicated().sum()),
        "transacoes_duplicadas": int(todos["trans_num"].duplicated().sum()),
        "cartoes_treino": int(tr["cc_num"].nunique()),
        "cartoes_teste": int(te["cc_num"].nunique()),
        "cartoes_em_comum": len(set(tr["cc_num"]) & set(te["cc_num"])),
        "cartoes_so_no_teste": int(exclusivos["cc_num"].nunique()),
        "transacoes_de_cartoes_so_no_teste": len(exclusivos),
        "fraudes_de_cartoes_so_no_teste": int(exclusivos[ALVO].sum()),
    }, so_no_teste.to_numpy()


def sensibilidade_aos_atributos(df):
    """Retreina o LightGBM com SMOTE retirando atributos que descrevem o portador."""
    treino, teste = df[df["conjunto"] == "treino"], df[df["conjunto"] == "teste"]
    y_te = teste[ALVO].to_numpy()
    resultado = {}
    for nome, (sem_num, sem_cat) in CENARIOS.items():
        num = [c for c in ATRIBUTOS_NUMERICOS if c not in sem_num]
        cat = [c for c in ATRIBUTOS_CATEGORICOS if c not in sem_cat]
        prep = criar_preprocessador(num, cat).fit(treino[num + cat])
        Xb, yb = reamostrar(prep.transform(treino[num + cat]), treino[ALVO].to_numpy(), "smote")
        modelo = criar_modelos(False, 1.0)["LightGBM"].fit(Xb, yb)
        score = modelo.predict_proba(prep.transform(teste[num + cat]))[:, 1]
        alerta = score >= 0.5
        vp = int((alerta & (y_te == 1)).sum())
        resultado[nome] = {
            "pr_auc": average_precision_score(y_te, score),
            "roc_auc": roc_auc_score(y_te, score),
            "recall": vp / int(y_te.sum()),
            "precisao": vp / max(int(alerta.sum()), 1),
        }
        print(f"  {nome}: PR-AUC={resultado[nome]['pr_auc']:.3f}", flush=True)
    return resultado


def incerteza_no_teste(so_no_teste):
    """Bootstrap sobre as transações de teste: cobre a amostra, não a variação do treino."""
    pred = pd.read_parquet(PASTA_ARTEFATOS / "smote" / "predicoes_teste.parquet")
    y = pred["y_true"].to_numpy()
    a, b = pred["LightGBM"].to_numpy(), pred["Random Forest"].to_numpy()
    rng = np.random.default_rng(SEMENTE)
    pr_a, dif = [], []
    for _ in range(N_BOOTSTRAP):
        i = rng.integers(0, len(y), len(y))
        pa = average_precision_score(y[i], a[i])
        pr_a.append(pa)
        dif.append(pa - average_precision_score(y[i], b[i]))
    return {
        "n_bootstrap": N_BOOTSTRAP,
        "pr_auc_lightgbm_ic95": np.percentile(pr_a, [2.5, 97.5]).tolist(),
        "diferenca_lightgbm_rf_ic95": np.percentile(dif, [2.5, 97.5]).tolist(),
        "pr_auc_lightgbm_cartoes_conhecidos": average_precision_score(y[~so_no_teste], a[~so_no_teste]),
    }


def main():
    print("Qualidade dos dados...")
    qualidade, so_no_teste = qualidade_dos_dados()
    df = construir_atributos(carregar_bruto())
    treino = df[df["conjunto"] == "treino"]
    print("Sensibilidade aos atributos...")
    saida = {
        "qualidade": qualidade,
        "taxa_por_hora_treino": treino.groupby("hora")[ALVO].mean().mul(100).round(3).to_dict(),
        "sensibilidade": sensibilidade_aos_atributos(df),
    }
    print("Bootstrap...")
    saida["incerteza"] = incerteza_no_teste(so_no_teste)
    arquivo = PASTA_ARTEFATOS / "analises_complementares.json"
    arquivo.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Gravado em {arquivo}")


if __name__ == "__main__":
    main()
