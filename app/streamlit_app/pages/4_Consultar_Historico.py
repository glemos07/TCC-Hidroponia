"""Consulta de historico de medições de uma planta."""
import plotly.express as px
import streamlit as st

from app_utils import header, sidebar_info, PH_MIN, PH_MAX, EC_MIN, EC_MAX
from src.database import fetch_df

sidebar_info()
header("Consultar historico", "Historico de medicoes agrupado por planta e sensor")

plants = fetch_df("SELECT id, name_plant FROM plants ORDER BY name_plant")
if plants.empty:
    st.warning("Sem plantas cadastradas.")
    st.stop()

id_plant = st.selectbox(
    "Selecione a planta",
    plants["id"],
    format_func=lambda i: plants.loc[plants["id"] == i, "name_plant"].values[0],
)
sensor_opt = st.selectbox("Sensor", ["pH", "EC", "Temperatura", "Umidade"])

df = fetch_df("""
    SELECT m.dt_register, sd.value, s.sensor_type
    FROM monitoring m
    JOIN sensor_data sd ON sd.id_monitoring = m.id
    JOIN sensor s ON s.id = sd.id_sensor
    WHERE m.id_plant = ? AND s.sensor_type = ?
    ORDER BY m.dt_register
""", params=(int(id_plant), sensor_opt))

if df.empty:
    st.info("Nenhuma leitura encontrada para essa combinacao.")
    st.stop()

df["dt_register"] = df["dt_register"].astype("datetime64[ns]")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Registros", len(df))
col2.metric("Media",  f"{df['value'].mean():.3f}")
col3.metric("Minimo", f"{df['value'].min():.3f}")
col4.metric("Maximo", f"{df['value'].max():.3f}")

fig = px.line(df, x="dt_register", y="value",
              title=f"Historico de {sensor_opt}",
              labels={"dt_register": "Data/Hora", "value": sensor_opt})
if sensor_opt == "pH":
    fig.add_hrect(y0=PH_MIN, y1=PH_MAX, fillcolor="green", opacity=0.08, line_width=0)
elif sensor_opt == "EC":
    fig.add_hrect(y0=EC_MIN, y1=EC_MAX, fillcolor="green", opacity=0.08, line_width=0)
st.plotly_chart(fig, use_container_width=True)

st.dataframe(df, use_container_width=True)
