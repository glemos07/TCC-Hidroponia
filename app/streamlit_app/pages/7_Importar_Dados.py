"""Inicializar o banco (schema + catalogos de referencia)."""
import streamlit as st

from app_utils import header, sidebar_info
from src.database import fetch_df

sidebar_info()
header("Inicializar banco", "Aplica o schema e carrega os catalogos de referencia")

st.markdown(
    "Este utilitario cria o banco com as **10 entidades do Apendice A** e "
    "carrega apenas os catalogos de referencia: status de cultivo, tipos de "
    "sensor e nutrientes.\n\n"
    "Nenhuma medicao e criada aqui. O historico do sistema vem dos sensores: "
    "importe-o na tela **10_Servidor_API**. Plantas, ambientes, responsaveis e "
    "colheitas sao cadastrados nas telas RF01, RF02 e RF05."
)

st.warning("Atencao: executar isto **apaga** todos os registros atuais do banco, "
           "inclusive as leituras ja importadas dos sensores.")

if st.button("Inicializar banco", type="primary"):
    from src.seed import seed
    with st.spinner("Aplicando schema e carregando catalogos..."):
        counts = seed()
    st.success("Banco inicializado.")
    st.json(counts)
    st.info("Proximo passo: importe as leituras dos sensores na tela 10_Servidor_API.")

st.markdown("### Volumes atuais do banco")
try:
    catalogos = {"farm_status", "sensor", "nutrient"}
    linhas = []
    for t in ["farm_status", "plants", "ambient", "users", "sensor", "monitoring",
              "sensor_data", "nutrient", "monitoring_nutrient", "farm"]:
        n = int(fetch_df(f"SELECT COUNT(*) AS n FROM {t}")["n"].iloc[0])
        linhas.append({
            "entidade": t,
            "registros": n,
            "origem": ("catalogo de referencia" if t in catalogos
                       else "sensores / cadastro manual"),
        })
    st.dataframe(linhas, use_container_width=True, hide_index=True)
except Exception as e:
    st.error(f"Banco ainda nao inicializado: {e}")
