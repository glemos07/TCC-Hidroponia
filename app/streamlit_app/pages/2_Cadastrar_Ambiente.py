"""Cadastro de ambiente."""
import streamlit as st

from app_utils import header, sidebar_info
from src.database import connect, fetch_df

sidebar_info()
header("Cadastrar ambiente", "Registra um setor/ambiente de cultivo")

with st.form("form_amb", clear_on_submit=True):
    nome = st.text_input("Nome do ambiente", placeholder="Ex.: Estufa NFT - Setor A")
    tipo = st.selectbox(
        "Tipo do sistema",
        ["Estufa NFT", "Estufa DFT", "Aeroponia", "Substrato inerte"],
    )
    desc = st.text_area("Descricao", placeholder="Detalhes do ambiente (opcional)")
    submit = st.form_submit_button("Cadastrar ambiente", type="primary")

if submit:
    if not nome:
        st.error("Informe o nome do ambiente.")
    else:
        with connect() as conn:
            cur = conn.execute(
                "INSERT INTO ambient (nm_ambient, tp_ambient, descriptions) VALUES (?, ?, ?)",
                (nome, tipo, desc or None),
            )
        st.success(f"Ambiente cadastrado com id={cur.lastrowid}.")

st.markdown("---")
st.markdown("### Ambientes cadastrados")
st.dataframe(
    fetch_df("SELECT id, nm_ambient, tp_ambient, descriptions FROM ambient ORDER BY id"),
    use_container_width=True,
)
