"""Painel de indicadores (Quadros 2, 4 e Grafico 1/2)."""
import plotly.express as px
import streamlit as st

from app_utils import header, sidebar_info, PH_MIN, PH_MAX, EC_MIN, EC_MAX
from src.database import fetch_df

sidebar_info()
header("Painel de indicadores", "Visao consolidada da fazenda hidroponica")

counts = fetch_df("""
    SELECT (SELECT COUNT(*) FROM plants)      AS plants,
           (SELECT COUNT(*) FROM ambient)     AS ambient,
           (SELECT COUNT(*) FROM monitoring)  AS monitoring,
           (SELECT COUNT(*) FROM sensor_data) AS sensor_data,
           (SELECT COUNT(*) FROM farm)        AS colheitas
""").iloc[0]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Plantas",       int(counts["plants"]))
c2.metric("Ambientes",     int(counts["ambient"]))
c3.metric("Monitoramentos", int(counts["monitoring"]))
c4.metric("Leituras",       int(counts["sensor_data"]))
c5.metric("Colheitas",      int(counts["colheitas"]))

st.markdown("### Quadro 4 - Estatisticas descritivas das variaveis monitoradas")
stats = fetch_df("""
    SELECT s.sensor_type AS variavel,
           MIN(sd.value) AS minimo,
           MAX(sd.value) AS maximo,
           AVG(sd.value) AS media
    FROM sensor_data sd
    JOIN sensor s ON s.id = sd.id_sensor
    WHERE s.sensor_type IN ('pH', 'EC', 'Temperatura', 'Umidade', 'Temperatura Ambiente')
    GROUP BY s.sensor_type
""")
st.dataframe(stats.round(3), use_container_width=True)

st.markdown("### Grafico 1 - Registros por variavel coletada")
dist = fetch_df("""
    SELECT s.sensor_type AS variavel, COUNT(*) AS registros
    FROM sensor_data sd
    JOIN sensor s ON s.id = sd.id_sensor
    GROUP BY s.sensor_type ORDER BY registros DESC
""")
st.plotly_chart(px.bar(dist, x="variavel", y="registros", text="registros"), use_container_width=True)

st.markdown("### Grafico 2 - Variacao de pH ao longo do ciclo")
ph = fetch_df("""
    SELECT m.dt_register, sd.value AS pH
    FROM sensor_data sd
    JOIN sensor s     ON s.id = sd.id_sensor
    JOIN monitoring m ON m.id = sd.id_monitoring
    WHERE s.sensor_type = 'pH'
    ORDER BY m.dt_register
""")
if not ph.empty:
    ph["dt_register"] = ph["dt_register"].astype("datetime64[ns]")
    fig = px.line(ph, x="dt_register", y="pH", title="Variacao de pH")
    fig.add_hrect(y0=PH_MIN, y1=PH_MAX, fillcolor="green", opacity=0.08, line_width=0)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### Producao por ciclo")
prod = fetch_df("""
    SELECT id, dt_harvest, weight_harvest AS peso_kg, time_growth AS ciclo_dias
    FROM farm ORDER BY dt_harvest
""")
if not prod.empty:
    st.plotly_chart(px.bar(prod, x="id", y="peso_kg", title="Peso colhido por ciclo"),
                    use_container_width=True)
