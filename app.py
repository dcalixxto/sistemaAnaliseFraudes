"""Dashboard Streamlit: análise exploratória e comparativo de modelos.

Execute antes `python -m src.treinar` e depois `streamlit run app.py`.

Como o Streamlit funciona: a cada interação (clique, slider, seleção) este
script inteiro roda de novo, de cima a baixo. Cada `st.algo(...)` desenha um
elemento na página, e os widgets (st.slider, st.selectbox...) devolvem o valor
que o usuário escolheu. Por isso as leituras de arquivo usam cache.
"""
import json

import numpy as np
import pandas as pd
import plotly.express as px          # gráficos prontos em 1 linha (barras, histogramas, mapas)
import plotly.graph_objects as go    # gráficos montados peça por peça (curvas, heatmap)
import plotly.io as pio
import streamlit as st
from plotly.subplots import make_subplots  # gráfico com dois eixos Y
from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_recall_curve, roc_curve,
)

from src.config import (
    ALVO, ARQ_DADOS_EDA, CATEGORIAS_PT, DIAS_SEMANA_PT, ESTRATEGIAS,
    MODELOS, PASTA_ARTEFATOS,
)

# Título da aba do navegador, ícone e layout largo. Deve ser o 1º comando st.*.
st.set_page_config(page_title="Análise de Fraudes", page_icon="💳", layout="wide")

# Todos os gráficos com vírgula decimal e ponto de milhar (padrão brasileiro).
pio.templates["pt_br"] = go.layout.Template(layout=dict(separators=",."))
pio.templates.default = "plotly+pt_br"

# Uma cor fixa por modelo em todos os gráficos, para o leitor associar cor e modelo.
CORES = {
    "Regressão Logística": "#2a78d6",
    "Naive Bayes": "#e8a33a",
    "Random Forest": "#1f9e6e",
    "LightGBM": "#c9466b",
}
# Um traço por modelo nas curvas: mantém as linhas distinguíveis em impressão preto e branco.
TRACOS = {
    "Regressão Logística": "dot",
    "Naive Bayes": "dashdot",
    "Random Forest": "dash",
    "LightGBM": "solid",
}
COR_FRAUDE, COR_LEGITIMA = "#c9466b", "#8a94a6"


# --------------------------------------------------------------------------- dados
# @st.cache_data guarda o resultado na memória após a 1ª chamada: os cliques
# seguintes não releem os arquivos (1,85 milhão de linhas).
@st.cache_data(show_spinner="Carregando transações...")
def carregar_eda():
    df = pd.read_parquet(ARQ_DADOS_EDA)
    # Coluna com a categoria traduzida; fillna mantém o original se faltar tradução.
    df["categoria"] = df["category"].map(CATEGORIAS_PT).fillna(df["category"])
    return df


def estrategias_disponiveis():
    """Só as estratégias já treinadas (cujo metricas.json existe)."""
    return [e for e in ESTRATEGIAS if (PASTA_ARTEFATOS / e / "metricas.json").exists()]


@st.cache_data
def carregar_resultados(estrategia):
    """Devolve (métricas do JSON, probabilidades do teste) de uma estratégia."""
    pasta = PASTA_ARTEFATOS / estrategia
    info = json.loads((pasta / "metricas.json").read_text(encoding="utf-8"))
    pred = pd.read_parquet(pasta / "predicoes_teste.parquet")
    return info, pred


# Nomes legíveis para os atributos no gráfico de importância.
NOMES_ATRIBUTOS = {
    "valor_log": "Valor (log)", "hora": "Hora do dia", "noturno": "Horário noturno (22h–3h)",
    "dia_semana": "Dia da semana", "idade": "Idade do portador", "distancia_km": "Distância ao comerciante",
    "city_pop_log": "População da cidade (log)", "lat": "Latitude do portador",
    "long": "Longitude do portador", "gender_F": "Gênero: F", "gender_M": "Gênero: M",
}
# Estilo CSS do melhor valor de cada coluna da tabela (legível no tema claro e escuro).
DESTAQUE = "font-weight: bold; background-color: rgba(31, 158, 110, 0.35)"


def milhar(n):
    """1852394 -> '1.852.394' (padrão brasileiro)."""
    return f"{n:,}".replace(",", ".")


def br(valor, casas=2):
    """1234.5 -> '1.234,50' (padrão brasileiro)."""
    return f"{valor:,.{casas}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def rotulo_atributo(nome):
    """'category_travel' -> 'Categoria: Viagem'; 'valor_log' -> 'Valor (log)'."""
    if nome.startswith("category_"):
        return "Categoria: " + CATEGORIAS_PT.get(nome.removeprefix("category_"), nome)
    return NOMES_ATRIBUTOS.get(nome, nome)


def taxa(serie):
    """Taxa de fraude em %: a média de uma coluna 0/1 é a proporção de uns."""
    return 100 * serie.mean()


# --------------------------------------------------------------------------- páginas
def pagina_eda():
    """Tela 1: análise exploratória dos dados históricos."""
    st.header("📊 Análise exploratória")
    df = carregar_eda()

    # Filtros na barra lateral ("with st.sidebar" desenha dentro dela).
    with st.sidebar:
        st.subheader("Filtros")
        conjunto = st.radio("Período", ["Todos", "Treino", "Teste"], horizontal=True)
        cats = st.multiselect("Categorias", sorted(df["categoria"].unique()), placeholder="Todas as categorias")
    # Indexação booleana: mantém só as linhas em que a condição é verdadeira.
    if conjunto != "Todos":
        df = df[df["conjunto"] == conjunto.lower()]
    if cats:
        df = df[df["categoria"].isin(cats)]
    if df.empty:
        st.warning("Nenhuma transação para os filtros escolhidos.")
        return

    # Indicadores (KPIs) em 5 colunas.
    fraudes = df[df[ALVO] == 1]
    c = st.columns(5)
    c[0].metric("Transações", milhar(len(df)))
    c[1].metric("Fraudes", milhar(len(fraudes)))
    c[2].metric("Taxa de fraude", f"{br(taxa(df[ALVO]))}%")
    # O 3º argumento (delta) mostra a comparação com as legítimas; delta_color="off" = cinza.
    c[3].metric("Ticket médio fraude", f"US$ {br(fraudes['amt'].mean())}",
                f"legítima: US$ {br(df.loc[df[ALVO] == 0, 'amt'].mean())}", delta_color="off")
    c[4].metric("Valor fraudado", f"US$ {br(fraudes['amt'].sum() / 1e6)} mi")

    st.caption("A taxa de fraude é ínfima: um modelo que responde sempre “legítima” teria "
               f"{br(100 - taxa(df[ALVO]))}% de acurácia e não detectaria nada — por isso a acurácia não é usada.")

    # --- Taxa por hora: groupby agrupa por hora; mean = taxa, sum = nº de fraudes.
    col1, col2 = st.columns(2)
    por_hora = df.groupby("hora")[ALVO].agg(["mean", "sum"]).reset_index()
    fig = px.bar(por_hora, x="hora", y=por_hora["mean"] * 100, color_discrete_sequence=[COR_FRAUDE],
                 labels={"y": "Taxa de fraude (%)", "hora": "Hora do dia"},
                 hover_data={"sum": True}, title="Taxa de fraude por hora do dia")
    col1.plotly_chart(fig, width="stretch")

    # --- Taxa por dia da semana; reindex(range(7)) garante os 7 dias em ordem.
    por_dia = df.groupby("dia_semana")[ALVO].mean().mul(100).reindex(range(7)).reset_index()
    por_dia["dia"] = DIAS_SEMANA_PT
    fig = px.bar(por_dia, x="dia", y=ALVO, color_discrete_sequence=[COR_FRAUDE],
                 labels={ALVO: "Taxa de fraude (%)", "dia": ""}, title="Taxa de fraude por dia da semana")
    col2.plotly_chart(fig, width="stretch")

    # --- Taxa por categoria (barras horizontais ordenadas pela taxa).
    col1, col2 = st.columns(2)
    por_cat = (df.groupby("categoria")[ALVO].agg(taxa="mean", fraudes="sum").reset_index()
                 .assign(taxa=lambda d: d["taxa"] * 100).sort_values("taxa"))
    fig = px.bar(por_cat, x="taxa", y="categoria", orientation="h", color_discrete_sequence=[COR_FRAUDE],
                 hover_data=["fraudes"], labels={"taxa": "Taxa de fraude (%)", "categoria": ""},
                 title="Categorias de comércio com maior risco")
    col1.plotly_chart(fig, width="stretch")

    legitimas = df[df[ALVO] == 0]
    # Amostra das legítimas (todas as fraudes + 50 mil legítimas) para os gráficos
    # de distribuição não pesarem no navegador; a forma da distribuição não muda.
    amostra = pd.concat([fraudes, legitimas.sample(min(len(legitimas), 50_000), random_state=0)])
    amostra["Classe"] = amostra[ALVO].map({0: "Legítima", 1: "Fraude"})
    # --- Boxplot do valor: linha central = mediana; caixa = 1º ao 3º quartil.
    fig = px.box(amostra, x="Classe", y="amt", color="Classe", log_y=True, points=False,
                 color_discrete_map={"Fraude": COR_FRAUDE, "Legítima": COR_LEGITIMA},
                 labels={"amt": "Valor (US$, escala log)", "Classe": ""},
                 title="Distribuição do valor da transação")
    col2.plotly_chart(fig, width="stretch")

    # --- Histogramas; histnorm="percent" compara formas mesmo com quantidades diferentes.
    col1, col2 = st.columns(2)
    fig = px.histogram(amostra, x="idade", color="Classe", histnorm="percent", barmode="overlay",
                       nbins=40, opacity=0.65, color_discrete_map={"Fraude": COR_FRAUDE, "Legítima": COR_LEGITIMA},
                       labels={"idade": "Idade do portador (anos)"}, title="Idade do portador por classe")
    col1.plotly_chart(fig, width="stretch")
    fig = px.histogram(amostra, x="distancia_km", color="Classe", histnorm="percent", barmode="overlay",
                       nbins=40, opacity=0.65, color_discrete_map={"Fraude": COR_FRAUDE, "Legítima": COR_LEGITIMA},
                       labels={"distancia_km": "Distância portador → comerciante (km)"},
                       title="Distância entre portador e comerciante")
    col2.plotly_chart(fig, width="stretch")
    dist = df.groupby(ALVO)["distancia_km"].mean()
    st.caption(f"Distância média — fraude: {br(dist.get(1, np.nan), 1)} km | legítima: {br(dist.get(0, np.nan), 1)} km.")

    # --- Mapas: calor das fraudes (coordenadas do comerciante) e taxa por estado.
    st.subheader("🗺️ Distribuição geográfica das fraudes")
    col1, col2 = st.columns(2)
    fig = px.density_map(fraudes, lat="merch_lat", lon="merch_long", radius=3, zoom=2.6,
                         center={"lat": 39, "lon": -97}, map_style="carto-positron",
                         color_continuous_scale="Reds", height=460)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_showscale=False)
    col1.plotly_chart(fig, width="stretch")
    por_uf = df.groupby("state")[ALVO].agg(taxa="mean", fraudes="sum", n="size").reset_index()
    por_uf["taxa"] *= 100
    # Estados com poucas transações distorcem a escala (ex.: DE tem 9, todas fraude = 100%).
    por_uf = por_uf[por_uf["n"] >= 1000]
    # choropleth = mapa com regiões pintadas; "USA-states" reconhece siglas (NY, TX...).
    fig = px.choropleth(por_uf, locations="state", locationmode="USA-states", scope="usa",
                        color="taxa", color_continuous_scale="Reds", hover_data=["fraudes"],
                        labels={"taxa": "Taxa (%)"}, height=460)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    col2.plotly_chart(fig, width="stretch")
    st.caption("À esquerda, concentração das fraudes pela posição do comerciante; à direita, taxa de fraude "
               "por estado do portador, sem os estados com menos de 1.000 transações.")

    # --- Evolução mensal: resample("MS") agrupa por mês (MS = início do mês).
    st.subheader("Evolução mensal")
    mensal = (df.set_index("trans_date_trans_time").resample("MS")[ALVO]
                .agg(["sum", "mean"]).reset_index())
    # Barras (quantidade) no eixo esquerdo e linha (taxa) no eixo direito.
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=mensal["trans_date_trans_time"], y=mensal["sum"], name="Nº de fraudes",
                marker_color=COR_FRAUDE, opacity=0.7)
    fig.add_scatter(x=mensal["trans_date_trans_time"], y=mensal["mean"] * 100, name="Taxa (%)",
                    line_color="#333", secondary_y=True)
    fig.update_layout(height=360, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")


def curva_amostrada(x, y, n=400):
    """Reduz uma curva a n pontos igualmente espaçados (suficiente para desenhar).

    As curvas ROC/PR têm um ponto por limiar possível (centenas de milhares).
    """
    idx = np.linspace(0, len(x) - 1, min(n, len(x))).astype(int)
    return x[idx], y[idx]


def pagina_modelos():
    """Tela 2: comparativo dos 4 modelos no conjunto de teste."""
    st.header("🤖 Comparativo dos modelos")
    disponiveis = estrategias_disponiveis()
    with st.sidebar:
        # format_func mostra a descrição, mas o valor devolvido é o código ("pesos"...).
        estrategia = st.selectbox("Estratégia de balanceamento", disponiveis, format_func=ESTRATEGIAS.get)
        # slider(rótulo, mínimo, máximo, valor inicial, passo)
        limiar = st.slider("Limiar de decisão", 0.01, 0.99, 0.50, 0.01,
                           help="Probabilidade acima da qual a transação é classificada como fraude. "
                                "Baixar o limiar aumenta o recall e diminui a precisão.")
    info, pred = carregar_resultados(estrategia)
    meta = info["meta"]
    y = pred["y_true"].to_numpy()  # rótulos reais do teste

    # [:10] corta "2019-01-01 00:00:18" para "2019-01-01".
    st.caption(
        f"Treino: {meta['periodo_treino'][0][:10]} a {meta['periodo_treino'][1][:10]} "
        f"({milhar(meta['n_treino_original'])} transações → {milhar(meta['n_treino_balanceado'])} "
        f"após balanceamento) · Teste: {meta['periodo_teste'][0][:10]} a {meta['periodo_teste'][1][:10]} "
        f"({milhar(meta['n_teste'])} transações, {milhar(meta['fraudes_teste'])} fraudes). "
        "O teste é temporalmente posterior ao treino e nunca é reamostrado."
    )

    # --- Tabela de métricas recalculada para o limiar escolhido.
    linhas = []
    for nome in MODELOS:
        s = pred[nome].to_numpy()  # probabilidades salvas no treino
        tn, fp, fn, tp = confusion_matrix(y, (s >= limiar).astype(int), labels=[0, 1]).ravel()
        # "x if cond else 0.0" evita divisão por zero quando não há alertas.
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        m = info["modelos"][nome]  # métricas que não dependem do limiar vêm do JSON
        linhas.append({
            "Modelo": nome, "Recall": rec, "Precisão": prec,
            "F1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "PR-AUC": m["pr_auc"], "ROC-AUC": m["roc_auc"],
            "Fraudes detectadas": int(tp), "Fraudes perdidas (FN)": int(fn),
            "Bloqueios indevidos (FP)": int(fp),
            "Treino (s)": m["tempo_treino_s"], "Latência (ms/transação)": m["latencia_ms_por_transacao"],
        })
    tabela = pd.DataFrame(linhas).set_index("Modelo")

    st.subheader(f"Métricas no conjunto de teste (limiar = {br(limiar)})")
    maiores = ["Recall", "Precisão", "F1", "PR-AUC", "ROC-AUC", "Fraudes detectadas"]  # quanto maior, melhor
    menores = ["Fraudes perdidas (FN)", "Bloqueios indevidos (FP)", "Treino (s)", "Latência (ms/transação)"]
    st.dataframe(
        tabela.style
        .highlight_max(subset=maiores, props=DESTAQUE)
        .highlight_min(subset=menores, props=DESTAQUE)
        # "|" junta dicionários: formato de número por coluna, no padrão brasileiro.
        .format({c: (lambda v: br(v, 3)) for c in ["Recall", "Precisão", "F1", "PR-AUC", "ROC-AUC"]}
                | {"Treino (s)": lambda v: br(v, 1), "Latência (ms/transação)": lambda v: br(v, 4)}
                | {c: milhar for c in ["Fraudes detectadas", "Fraudes perdidas (FN)", "Bloqueios indevidos (FP)"]}),
        width="stretch",
    )
    # idxmax devolve o nome da linha (modelo) com o maior valor.
    melhor_rec = tabela["Recall"].idxmax()
    melhor_prauc = tabela["PR-AUC"].idxmax()
    st.info(f"**Maior recall:** {melhor_rec} ({br(100 * tabela.loc[melhor_rec, 'Recall'], 1)}%).  "
            f"**Melhor PR-AUC** (desempenho independente do limiar): {melhor_prauc} "
            f"({br(tabela.loc[melhor_prauc, 'PR-AUC'], 3)}). "
            "Recall alto com precisão muito baixa significa bloquear muitos clientes legítimos.")

    # --- Curva Precision-Recall dos 4 modelos sobrepostos.
    col1, col2 = st.columns(2)
    fig = go.Figure()
    for nome in MODELOS:
        p, r, _ = precision_recall_curve(y, pred[nome])  # _ = limiares (não usados)
        r, p = curva_amostrada(r, p)
        fig.add_scatter(x=r, y=p, name=f"{nome} (AP={br(average_precision_score(y, pred[nome]), 3)})",
                        line=dict(color=CORES[nome], width=2, dash=TRACOS[nome]))
    # Um classificador aleatório tem precisão = taxa de fraude em qualquer recall.
    fig.add_hline(y=y.mean(), line_dash="dot", line_color="gray", annotation_text="aleatório")
    fig.update_layout(title="Curva Precision-Recall", xaxis_title="Recall", yaxis_title="Precisão",
                      height=430, legend=dict(orientation="h", y=-0.2))
    col1.plotly_chart(fig, width="stretch")

    # --- Curva ROC dos 4 modelos; a diagonal é o classificador aleatório.
    fig = go.Figure()
    for nome in MODELOS:
        fpr, tpr, _ = roc_curve(y, pred[nome])  # taxa de falsos e de verdadeiros positivos
        fpr, tpr = curva_amostrada(fpr, tpr)
        fig.add_scatter(x=fpr, y=tpr, name=f"{nome} (AUC={br(info['modelos'][nome]['roc_auc'], 3)})",
                        line=dict(color=CORES[nome], width=2, dash=TRACOS[nome]))
    fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dot", color="gray"), showlegend=False)
    fig.update_layout(title="Curva ROC", xaxis_title="Taxa de falsos positivos",
                      yaxis_title="Taxa de verdadeiros positivos (recall)", height=430,
                      legend=dict(orientation="h", y=-0.2))
    col2.plotly_chart(fig, width="stretch")
    st.caption("Com classes tão desbalanceadas a curva ROC tende a parecer otimista; "
               "a curva Precision-Recall é a referência principal deste comparativo.")

    # --- Matrizes de confusão lado a lado (zip percorre colunas e modelos juntos).
    st.subheader("Matrizes de confusão")
    cols = st.columns(4)
    for col, nome in zip(cols, MODELOS):
        cm = confusion_matrix(y, (pred[nome] >= limiar).astype(int), labels=[0, 1])
        texto = [[milhar(v) for v in linha] for linha in cm]  # número exibido em cada célula
        # A cor usa log1p: sem isso os ~550 mil verdadeiros negativos apagariam o resto.
        fig = go.Figure(go.Heatmap(
            z=np.log1p(cm), x=["Prev. legítima", "Prev. fraude"], y=["Real legítima", "Real fraude"],
            text=texto, texttemplate="%{text}", colorscale="Blues", showscale=False, hoverinfo="skip"))
        fig.update_layout(title=dict(text=nome, font=dict(size=14)), height=280,
                          margin=dict(l=10, r=10, t=40, b=10), yaxis_autorange="reversed")
        col.plotly_chart(fig, width="stretch")

    # --- Barras agrupadas das métricas. melt transforma colunas em linhas
    #     (formato "longo") que o px.bar agrupa por métrica.
    col1, col2 = st.columns(2)
    long = tabela[["Recall", "Precisão", "F1", "PR-AUC"]].reset_index().melt(id_vars="Modelo")
    fig = px.bar(long, x="variable", y="value", color="Modelo", barmode="group",
                 color_discrete_map=CORES, labels={"variable": "", "value": ""},
                 title="Métricas lado a lado")
    fig.update_layout(height=400, legend=dict(orientation="h", y=-0.15))
    col1.plotly_chart(fig, width="stretch")
    # Tempo em escala log: os valores vão de ~0,2 s a ~60 s.
    fig = px.bar(tabela.reset_index(), x="Modelo", y="Treino (s)", color="Modelo",
                 color_discrete_map=CORES, log_y=True, title="Tempo de treinamento (escala log)",
                 hover_data=["Latência (ms/transação)"])
    fig.update_layout(height=400, showlegend=False)
    col2.plotly_chart(fig, width="stretch")

    # --- Importância dos atributos (só modelos que têm a medida).
    st.subheader("Importância dos atributos")
    nome = st.selectbox("Modelo", [m for m in MODELOS if info["importancias"].get(m)])
    # Remove o prefixo "num__"/"cat__" que o ColumnTransformer põe nos nomes.
    imp = pd.Series(info["importancias"][nome]).rename(lambda s: s.split("__", 1)[1])
    imp.index = imp.index.map(rotulo_atributo)
    # 15 maiores em valor absoluto; [::-1] inverte para o maior ficar no topo.
    top = imp.reindex(imp.abs().sort_values(ascending=False).index).head(15)[::-1]
    if nome == "Regressão Logística":
        # Cor pelo sinal do coeficiente.
        fig = px.bar(x=top.values, y=top.index, orientation="h",
                     color=np.where(top.values > 0, "Aumenta o risco", "Reduz o risco"),
                     color_discrete_map={"Aumenta o risco": COR_FRAUDE, "Reduz o risco": "#2a78d6"},
                     labels={"x": "Coeficiente (atributos padronizados)", "y": "", "color": ""})
        st.caption("Coeficientes da regressão: exp(coef) é a razão de chances (odds ratio) "
                   "para +1 desvio-padrão do atributo.")
    else:
        fig = px.bar(x=top.values, y=top.index, orientation="h", color_discrete_sequence=[CORES[nome]],
                     labels={"x": "Importância relativa", "y": ""})
    fig.update_layout(height=480)
    st.plotly_chart(fig, width="stretch")
    st.caption("As medidas não são comparáveis entre modelos: no Random Forest, redução de impureza; no LightGBM, "
               "número de cortes, que favorece atributos com muitos valores. O Naive Bayes Gaussiano não produz "
               "uma medida nativa de importância.")

    # --- Comparação entre estratégias (se mais de uma foi treinada).
    if len(disponiveis) > 1:
        st.subheader("Comparação entre estratégias de balanceamento (limiar 0,5)")
        linhas = []
        for est in disponiveis:
            inf, _ = carregar_resultados(est)
            for nome, m in inf["modelos"].items():
                linhas.append({"Estratégia": est, "Modelo": nome, "Recall": m["recall"],
                               "Precisão": m["precisao"], "F1": m["f1"], "PR-AUC": m["pr_auc"]})
        comp = pd.DataFrame(linhas)
        metrica = st.radio("Métrica", ["Recall", "Precisão", "F1", "PR-AUC"], horizontal=True)
        fig = px.bar(comp, x="Modelo", y=metrica, color="Estratégia", barmode="group",
                     color_discrete_sequence=["#2a78d6", "#e8a33a", "#1f9e6e"])
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- main
def main():
    st.sidebar.title("💳 Análise de Fraudes")
    # Sem artefatos não há o que mostrar: explica o que rodar e interrompe (st.stop).
    if not ARQ_DADOS_EDA.exists() or not estrategias_disponiveis():
        st.error("Artefatos não encontrados. Coloque os CSVs do Kaggle em `data/` e rode "
                 "`python -m src.treinar` antes de abrir o dashboard.")
        st.stop()
    # Menu lateral; usamos radio em vez de st.tabs porque as abas executariam as
    # 2 telas a cada clique, e aqui só a tela escolhida roda.
    pagina = st.sidebar.radio("Navegação", ["Análise exploratória", "Comparativo de modelos"])
    st.sidebar.divider()
    # Dicionário nome -> função; [pagina]() chama só a função da tela escolhida.
    {"Análise exploratória": pagina_eda,
     "Comparativo de modelos": pagina_modelos}[pagina]()
    st.sidebar.divider()
    st.sidebar.caption("TCC — Daniel Calixto Lima · UNIPAC\nDados sintéticos: Sparkov (Kaggle)")


main()
