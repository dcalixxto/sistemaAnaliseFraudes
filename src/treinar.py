"""Treina e avalia os 4 classificadores e salva os artefatos lidos pelo dashboard.

Uso:
    python -m src.treinar                          # estratégia "pesos"
    python -m src.treinar --estrategia undersampling
    python -m src.treinar --estrategia todas       # roda as 3 estratégias
"""
import argparse   # lê argumentos da linha de comando (--estrategia ...)
import json       # grava as métricas em texto estruturado
import time       # cronômetro de alta precisão (perf_counter)

import joblib     # salva/carrega objetos Python (modelos) em arquivo
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline  # encadeia pré-processamento + modelo

from src.config import (
    ALVO, ARQ_DADOS_EDA, ESTRATEGIAS, PASTA_ARTEFATOS,
)
from src.features import carregar_bruto, construir_atributos, separar_xy
from src.modelos import criar_modelos, criar_preprocessador, pesos_amostra, reamostrar

# Colunas salvas para a tela de análise exploratória do dashboard.
COLUNAS_EDA = [
    "trans_date_trans_time", "conjunto", "category", "amt", "gender", "state",
    "lat", "long", "merch_lat", "merch_long", "hora", "dia_semana", "idade",
    "distancia_km", ALVO,
]


def calcular_metricas(y_true, y_score, limiar=0.5):
    """Compara o real (y_true) com as probabilidades do modelo (y_score)."""
    # Probabilidade -> decisão: >= limiar vira alerta de fraude (1).
    y_pred = (y_score >= limiar).astype(int)
    # Matriz 2x2 achatada na ordem: verdadeiro negativo, falso positivo,
    # falso negativo, verdadeiro positivo.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        # Métricas sobre as DECISÕES (dependem do limiar).
        # zero_division=0: devolve 0 em vez de erro se não houver nenhum alerta.
        "precisao": precision_score(y_true, y_pred, zero_division=0),  # VP/(VP+FP)
        "recall": recall_score(y_true, y_pred, zero_division=0),       # VP/(VP+FN)
        "f1": f1_score(y_true, y_pred, zero_division=0),               # média harmônica
        # Métricas sobre as PROBABILIDADES (avaliam todos os limiares de uma vez).
        "pr_auc": average_precision_score(y_true, y_score),  # área sob a curva Precision-Recall
        "roc_auc": roc_auc_score(y_true, y_score),           # área sob a curva ROC
        # int(...) converte do tipo NumPy para int comum, que o JSON aceita.
        "vp": int(tp), "fp": int(fp), "fn": int(fn), "vn": int(tn),
    }


def importancias(nome, modelo, nomes_atributos):
    """Importância de cada atributo, quando o modelo oferece uma medida nativa."""
    if hasattr(modelo, "feature_importances_"):
        # Árvores (Random Forest e LightGBM): contribuição nos cortes das árvores.
        valores = modelo.feature_importances_.astype(float)
        valores = valores / valores.sum()  # normaliza para somar 1 (proporção)
    elif hasattr(modelo, "coef_"):
        # Regressão Logística: coeficientes; o sinal indica aumento (+) ou redução (-) do risco.
        valores = modelo.coef_[0]
    else:
        return None  # Naive Bayes Gaussiano não tem medida nativa
    # zip junta nome e valor; float(...) converte para número comum (JSON).
    return dict(zip(nomes_atributos, map(float, valores)))


def preparar_dados():
    """Carrega a base uma única vez e grava os arquivos auxiliares do dashboard."""
    print("Carregando dados brutos...")
    df = construir_atributos(carregar_bruto())
    PASTA_ARTEFATOS.mkdir(exist_ok=True)  # cria artifacts/ se não existir
    # Parquet: formato colunar compactado, lido ~15x mais rápido que CSV.
    df[COLUNAS_EDA].to_parquet(ARQ_DADOS_EDA, index=False)
    # :, = separador de milhar; :.3% = percentual com 3 casas.
    print(f"  {len(df):,} transações | fraudes: {df[ALVO].sum():,} ({df[ALVO].mean():.3%})")
    return df


def treinar_estrategia(df, estrategia):
    """Treina, avalia e salva os 4 modelos para uma estratégia de balanceamento."""
    print(f"\n=== Estratégia: {ESTRATEGIAS[estrategia]} ===")
    pasta = PASTA_ARTEFATOS / estrategia
    # parents=True cria as pastas intermediárias; exist_ok=True não reclama se já existir.
    (pasta / "modelos").mkdir(parents=True, exist_ok=True)

    # 1) Separa treino e teste pela coluna "conjunto" (divisão temporal).
    X_tr, y_tr = separar_xy(df[df["conjunto"] == "treino"])
    X_te, y_te = separar_xy(df[df["conjunto"] == "teste"])

    # 2) Ajusta (fit) o pré-processador SÓ no treino: média/desvio e categorias
    #    vêm do treino. Usar o teste aqui seria vazamento de dados.
    prep = criar_preprocessador().fit(X_tr)
    # Nomes das 25 colunas resultantes, ex.: "num__valor_log", "cat__category_travel".
    nomes_atributos = list(prep.get_feature_names_out())
    # Aplica a MESMA transformação aos dois conjuntos.
    Xp_tr, Xp_te = prep.transform(X_tr), prep.transform(X_te)

    # 3) Balanceia só o treino e cronometra (o tempo entra no custo do modelo).
    t0 = time.perf_counter()
    Xb, yb = reamostrar(Xp_tr, y_tr.to_numpy(), estrategia)
    tempo_reamostragem = time.perf_counter() - t0
    print(f"  Treino após balanceamento: {len(yb):,} linhas, {int(yb.sum()):,} fraudes")

    usar_pesos = estrategia == "pesos"
    # Razão legítimas/fraudes para o scale_pos_weight do LightGBM.
    # max(..., 1) evita divisão por zero.
    razao = (yb == 0).sum() / max((yb == 1).sum(), 1)
    # resultados: métricas; predicoes: probabilidades no teste; imp: importâncias.
    resultados, predicoes, imp = {}, {"y_true": y_te.to_numpy()}, {}

    # 4) Para cada um dos 4 modelos...
    for nome, modelo in criar_modelos(usar_pesos, razao).items():
        print(f"  -> {nome}...", end=" ", flush=True)  # flush: mostra na hora
        # O Naive Bayes não tem class_weight; na estratégia "pesos" recebe o peso
        # de cada linha. **fit_kwargs desempacota o dicionário como argumentos.
        fit_kwargs = {"sample_weight": pesos_amostra(yb)} if (usar_pesos and nome == "Naive Bayes") else {}

        # 4a) Treino cronometrado.
        t0 = time.perf_counter()
        modelo.fit(Xb, yb, **fit_kwargs)
        tempo_treino = time.perf_counter() - t0

        # 4b) Previsão no teste cronometrada. predict_proba devolve 2 colunas
        #     [P(legítima), P(fraude)]; [:, 1] pega a probabilidade de fraude.
        t0 = time.perf_counter()
        score = modelo.predict_proba(Xp_te)[:, 1]
        tempo_pred = time.perf_counter() - t0

        # 4c) Métricas + tempos (latência = ms por transação).
        m = calcular_metricas(y_te.to_numpy(), score)
        m.update(tempo_treino_s=tempo_treino + tempo_reamostragem,
                 tempo_predicao_s=tempo_pred,
                 latencia_ms_por_transacao=1000 * tempo_pred / len(score))
        resultados[nome] = m
        # float32 ocupa metade do espaço de float64 e basta para gráficos.
        predicoes[nome] = score.astype(np.float32)
        imp[nome] = importancias(nome, modelo, nomes_atributos)

        # 4d) Salva o Pipeline completo (pré-processamento + modelo) para que o
        #     modelo treinado possa ser reutilizado sem retreinar (ex.: aplicar a
        #     novas transações). compress=3 reduz o tamanho do arquivo.
        joblib.dump(Pipeline([("prep", prep), ("modelo", modelo)]),
                    pasta / "modelos" / f"{nome}.joblib", compress=3)
        print(f"recall={m['recall']:.3f} precisão={m['precisao']:.3f} "
              f"F1={m['f1']:.3f} PR-AUC={m['pr_auc']:.3f} ({tempo_treino:.1f}s)")

    # 5) Metadados exibidos no dashboard (tamanhos e períodos).
    meta = {
        "estrategia": estrategia,
        "descricao": ESTRATEGIAS[estrategia],
        "n_treino_original": int(len(y_tr)),
        "n_treino_balanceado": int(len(yb)),
        "fraudes_treino_balanceado": int(yb.sum()),
        "n_teste": int(len(y_te)),
        "fraudes_teste": int(y_te.sum()),
        "periodo_treino": [str(df.loc[df.conjunto == "treino", "trans_date_trans_time"].min()),
                           str(df.loc[df.conjunto == "treino", "trans_date_trans_time"].max())],
        "periodo_teste": [str(df.loc[df.conjunto == "teste", "trans_date_trans_time"].min()),
                          str(df.loc[df.conjunto == "teste", "trans_date_trans_time"].max())],
    }
    # 6) Grava métricas em JSON (ensure_ascii=False mantém os acentos legíveis).
    (pasta / "metricas.json").write_text(
        json.dumps({"meta": meta, "modelos": resultados, "importancias": imp},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    # 7) Grava as probabilidades do teste: com elas o dashboard recalcula curvas
    #    e matrizes para qualquer limiar sem rodar os modelos de novo.
    pd.DataFrame(predicoes).to_parquet(pasta / "predicoes_teste.parquet", index=False)


def main():
    # Lê o argumento --estrategia; a docstring do módulo vira o texto de ajuda (-h).
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--estrategia", choices=[*ESTRATEGIAS, "todas"], default="pesos")
    args = parser.parse_args()

    df = preparar_dados()  # carrega uma vez e reaproveita em todas as estratégias
    estrategias = list(ESTRATEGIAS) if args.estrategia == "todas" else [args.estrategia]
    for est in estrategias:
        treinar_estrategia(df, est)
    print(f"\nArtefatos salvos em {PASTA_ARTEFATOS}. Rode: streamlit run app.py")


# Só executa main() quando o arquivo é rodado diretamente (python -m src.treinar),
# não quando é importado por outro módulo.
if __name__ == "__main__":
    main()
