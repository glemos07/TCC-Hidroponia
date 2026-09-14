"""Carrega séries temporais de pH, CE, temperatura e umidade.

Prefere o CSV `timeseries_full.csv` (série longa, para treino de ML) quando
disponível; caso contrário lê do SQLite (Quadro 2 do doc, série truncada).
"""
from __future__ import annotations

import pandas as pd

from src.config import DATA_DIR
from src.database import connect


SENSOR_MAP = {
    "pH":          1,
    "EC":          2,
    "temperature": 3,
    "humidity":    4,
}


def load_timeseries(prefer_full: bool = True) -> pd.DataFrame:
    """
    Ordem de preferencia:
      1. CSV do Kaggle em data/kaggle/ (via kaggle_loader) - dado real
      2. CSV timeseries_full.csv (gerador sintetico com serie longa para ML)
      3. Banco SQLite (serie truncada, alinhada ao Quadro 2 do doc)
    """
    if prefer_full:
        # 1) Kaggle real
        try:
            from src.kaggle_loader import load_kaggle_csv
            r = load_kaggle_csv()
            if r is not None:
                df, _ = r
                return df[["timestamp", "pH", "EC", "temperature", "humidity"]]
        except Exception:
            pass
        # 2) sintetico longo
        csv_full = DATA_DIR / "timeseries_full.csv"
        if csv_full.exists():
            df = pd.read_csv(csv_full)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df[["timestamp", "pH", "EC", "temperature", "humidity"]]
    # 3) SQLite
    return _load_from_db()


def _load_from_db() -> pd.DataFrame:
    """Retorna DataFrame com timestamp e colunas pH, EC, temperature, humidity."""
    sql = """
    SELECT m.id            AS monitoring_id,
           m.dt_register   AS timestamp,
           s.sensor_type   AS sensor_type,
           sd.value        AS value
    FROM   monitoring m
    JOIN   sensor_data sd ON sd.id_monitoring = m.id
    JOIN   sensor s       ON s.id = sd.id_sensor
    WHERE  s.sensor_type IN ('pH', 'EC', 'Temperatura', 'Umidade')
    """
    with connect() as conn:
        df = pd.read_sql_query(sql, conn)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["sensor_type"] = df["sensor_type"].map({
        "pH":           "pH",
        "EC":           "EC",
        "Temperatura":  "temperature",
        "Umidade":      "humidity",
    })
    wide = (df.pivot_table(index="timestamp",
                           columns="sensor_type",
                           values="value",
                           aggfunc="mean")
              .sort_index()
              .reset_index())
    for col in ["pH", "EC", "temperature", "humidity"]:
        if col not in wide.columns:
            wide[col] = float("nan")
    return wide[["timestamp", "pH", "EC", "temperature", "humidity"]].dropna().reset_index(drop=True)


def train_test_split_series(df: pd.DataFrame, test_frac: float = 0.20):
    """Hold-out: últimos `test_frac` para teste, conforme metodologia 4.6."""
    n_test = max(1, int(len(df) * test_frac))
    train = df.iloc[:-n_test].reset_index(drop=True)
    test  = df.iloc[-n_test:].reset_index(drop=True)
    return train, test
