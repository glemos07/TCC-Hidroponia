"""Importar dados iniciais (executa seed do dataset)."""
import streamlit as st

from app_utils import header, sidebar_info
from src.database import fetch_df

sidebar_info()
header("Importar dados iniciais", "Executa o seed do dataset (schema + carga)")

st.markdown(
    "Este utilitario cria/recria o banco e carrega o dataset. "
    "Corresponde ao **Quadro 1 - Resumo do tratamento dos dados** e ao "
    "**Quadro 2 - Entidades do modelo relacional**."
)

st.warning("Atencao: executar o seed *apaga* os registros atuais do banco.")

if st.button("Executar seed", type="primary"):
    from src.seed import seed
    with st.spinner("Aplicando schema e carregando dataset..."):
        counts = seed()
    st.success("Seed concluido.")
    st.json(counts)

st.markdown("### Volumes atuais do banco")
try:
    linhas = []
    for t in ["farm_status","plants","ambient","users","sensor","monitoring",
              "sensor_data","nutrient","monitoring_nutrient","farm"]:
        n = fetch_df(f"SELECT COUNT(*) AS n FROM {t}")["n"].iloc[0]
        linhas.append({"entidade": t, "registros": int(n)})
    st.dataframe(linhas, use_container_width=True)
except Exception as e:
    st.error(f"Banco ainda nao inicializado: {e}")
