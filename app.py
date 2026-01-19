
---

## 3) app.py (pronto para Hugging Face)

Crie `app.py` com este conteúdo (ele é mais robusto que o seu anterior e evita avisos):

```python
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Risco & Retorno (Dados Públicos)", layout="wide")

@st.cache_data
def load_data():
    prices = pd.read_parquet("artifacts/prices.parquet")
    rets = pd.read_parquet("artifacts/returns.parquet")
    macro = pd.read_parquet("artifacts/macro_daily.parquet")
    ml_df = pd.read_parquet("artifacts/ml_dataset.parquet")
    with open("artifacts/metrics.json", "r") as f:
        metrics = json.load(f)
    return prices, rets, macro, ml_df, metrics

@st.cache_resource
def load_models():
    ridge = joblib.load("artifacts/model_ridge.pkl")
    rf = joblib.load("artifacts/model_rf.pkl")
    return ridge, rf

prices, rets, macro, ml_df, metrics = load_data()
ridge, rf = load_models()

st.title("Risco e Retorno no Mercado Financeiro (dados públicos)")
st.caption("Projeto educacional. Não é recomendação de investimento.")

tickers = list(rets.columns)

st.sidebar.header("Controles")
ticker = st.sidebar.selectbox("Ativo", tickers, index=tickers.index(metrics.get("target_ticker", tickers[0])) if metrics.get("target_ticker") in tickers else 0)

tab1, tab2, tab3, tab4 = st.tabs(["Curva de riqueza", "Drawdown", "Correlação", "ML (Risco futuro)"])

with tab1:
    st.subheader("Curva de riqueza (retorno acumulado)")
    wealth = (1 + rets[[ticker]].fillna(0)).cumprod()
    fig = px.line(wealth, x=wealth.index, y=ticker)
    fig.update_layout(hovermode="x unified", xaxis_title="Data", yaxis_title="Crescimento normalizado")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Drawdown")
    wealth = (1 + rets[[ticker]].fillna(0)).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1
    fig = px.line(dd, x=dd.index, y=ticker)
    fig.update_layout(hovermode="x unified", xaxis_title="Data", yaxis_title="Drawdown")
    st.plotly_chart(fig, use_container_width=True)
    st.write("Máximo drawdown no período:", float(dd.min().values[0]))

with tab3:
    st.subheader("Correlação (retornos diários)")
    corr = rets.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader(f"Estimativa de volatilidade futura (21 dias) — alvo: {metrics.get('target_ticker','SPY')}")
    st.caption("Modelos treinados no dataset do projeto. Interpretar como estudo empírico.")

    if "y_vol_21_forward" not in ml_df.columns:
        st.error("Coluna y_vol_21_forward não encontrada em artifacts/ml_dataset.parquet.")
    else:
        X = ml_df.drop(columns=["y_vol_21_forward"])
        y = ml_df["y_vol_21_forward"]

        pred_ridge = ridge.predict(X)
        pred_rf = rf.predict(X)

        dfp = pd.DataFrame(
            {"y_true": y, "ridge": pred_ridge, "random_forest": pred_rf},
            index=ml_df.index
        ).tail(900)

        fig = px.line(dfp, x=dfp.index, y=["y_true", "ridge", "random_forest"])
        fig.update_layout(hovermode="x unified", xaxis_title="Data", yaxis_title="Volatilidade anualizada")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Métricas (holdout final)")
        st.json(metrics)

st.markdown("---")
st.markdown(
    "**Limitações:** resultados históricos não garantem resultados futuros. "
    "Séries macro têm frequências diferentes e podem ter revisões. "
    "Projeto educacional — não é recomendação de investimento."
)
