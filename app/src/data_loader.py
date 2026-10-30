"""Carrega séries temporais multivariadas para os modelos preditivos.

Duas fontes, sem nenhum dado gerado artificialmente:

  - "api": telemetria real do ESP32 ingerida por `src.api_ingest` e lida do
    banco (pH, CE, temperatura da solucao, temperatura do ar). E a fonte de
    operacao do sistema.
  - "kaggle": dataset publico de hidroponia lido direto do CSV em
    `data/kaggle/`, mantido apenas como base de comparacao do Quadro 6 /
    hipotese H1. Nao entra no banco: o banco guarda somente o que os
    sensores mediram.

O vetor de entrada de cada fonte esta em `config.FEATURE_SETS`.
"""
from __future__ import annotations

import pandas as pd

from src.config import SENSOR_TO_FEATURE, features_de
from src.database import connect


def load_timeseries(fonte: str = "kaggle") -> pd.DataFrame:
    """Serie multivariada da fonte escolhida, pronta para ARIMA/LSTM."""
    if fonte == "api":
        return load_timeseries_api()
    if fonte == "kaggle":
        return _load_kaggle()
    raise ValueError(f"Fonte desconhecida: {fonte!r}. Use 'api' ou 'kaggle'.")


def _load_kaggle() -> pd.DataFrame:
    """Serie do dataset publico, lida do CSV (nunca do banco)."""
    from src.kaggle_loader import load_kaggle_csv

    resultado = load_kaggle_csv()
    if resultado is None:
        raise FileNotFoundError(
            "Nenhum CSV encontrado em data/kaggle/. Baixe o dataset do Kaggle "
            "para comparar com o Quadro 6, ou use a fonte 'api' para treinar "
            "sobre os dados dos sensores."
        )
    df, _relatorio = resultado
    return df[["timestamp"] + features_de("kaggle")]


def _tabela_existe(conn, nome: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (nome,)
    ).fetchone() is not None


def plantas_api() -> pd.DataFrame:
    """
    Plantas cujo historico veio da API, da mais para a menos amostrada.
    Colunas: id_plant (chave em PLANTS), id_planta (chave no ESP32),
    nome_planta, leituras.
    """
    vazio = pd.DataFrame(columns=["id_plant", "id_planta", "nome_planta", "leituras"])
    with connect() as conn:
        if not _tabela_existe(conn, "api_planta"):
            return vazio
        return pd.read_sql_query(
            """
            SELECT ap.id_plant         AS id_plant,
                   ap.id_planta        AS id_planta,
                   ap.nome_planta      AS nome_planta,
                   COUNT(l.id_remoto)  AS leituras
            FROM   api_planta ap
            LEFT   JOIN api_leitura l ON l.id_planta = ap.id_planta
            GROUP  BY ap.id_planta
            ORDER  BY leituras DESC
            """,
            conn,
        )


def load_timeseries_api(id_plant: int | None = None) -> pd.DataFrame:
    """
    Serie da telemetria real ingerida da API.

    Sem `id_plant`, usa a planta com mais leituras - treinar um unico modelo
    concatenando plantas diferentes criaria saltos artificiais na serie.
    """
    features = features_de("api")
    if id_plant is None:
        plantas = plantas_api()
        if plantas.empty or int(plantas["leituras"].iloc[0]) == 0:
            return pd.DataFrame(columns=["timestamp"] + features)
        id_plant = int(plantas["id_plant"].iloc[0])

    return load_plant_features(int(id_plant), features)


def load_plant_features(id_plant: int, features: list[str],
                        limite: int | None = None) -> pd.DataFrame:
    """
    Historico de uma planta em formato largo: uma coluna por feature.
    Compartilhado entre o treino (api) e a inferencia (src.predict).

    `limite` restringe aos N monitoramentos mais recentes. A inferencia so
    precisa da ultima janela, e carregar o historico inteiro para usar 240
    linhas custa caro e piora a cada dia de coleta.
    """
    if limite:
        sql = """
        SELECT m.dt_register AS timestamp,
               s.sensor_type AS sensor_type,
               sd.value      AS value
        FROM  (SELECT id, dt_register FROM monitoring
               WHERE id_plant = ? ORDER BY dt_register DESC, id DESC LIMIT ?) m
        JOIN   sensor_data sd ON sd.id_monitoring = m.id
        JOIN   sensor s       ON s.id = sd.id_sensor
        ORDER  BY m.dt_register
        """
        params: tuple = (int(id_plant), int(limite))
    else:
        sql = """
        SELECT m.dt_register AS timestamp,
               s.sensor_type AS sensor_type,
               sd.value      AS value
        FROM   monitoring m
        JOIN   sensor_data sd ON sd.id_monitoring = m.id
        JOIN   sensor s       ON s.id = sd.id_sensor
        WHERE  m.id_plant = ?
        ORDER  BY m.dt_register
        """
        params = (int(id_plant),)

    with connect() as conn:
        df = pd.read_sql_query(sql, conn, params=params)

    if df.empty:
        return pd.DataFrame(columns=["timestamp"] + features)

    df["feature"] = df["sensor_type"].map(SENSOR_TO_FEATURE)
    df = df[df["feature"].isin(features)]
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    wide = (df.pivot_table(index="timestamp", columns="feature",
                           values="value", aggfunc="mean")
              .sort_index()
              .reset_index())
    for col in features:
        if col not in wide.columns:
            wide[col] = float("nan")
    return wide[["timestamp"] + features].dropna().reset_index(drop=True)


def train_test_split_series(df: pd.DataFrame, test_frac: float = 0.20):
    """Hold-out: últimos `test_frac` para teste, conforme metodologia 4.6."""
    n_test = max(1, int(len(df) * test_frac))
    train = df.iloc[:-n_test].reset_index(drop=True)
    test  = df.iloc[-n_test:].reset_index(drop=True)
    return train, test
