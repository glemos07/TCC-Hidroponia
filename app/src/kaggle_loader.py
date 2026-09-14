"""
Adaptador para o dataset real do Kaggle (itsmonir31/hydroponics-datasets).

Este loader:
1. Procura qualquer CSV em `data/kaggle/`
2. Detecta colunas por sinonimos (case-insensitive)
3. Aplica limpeza (nulos, duplicatas) conforme Quadro 1 do documento
4. Mapeia para o esquema do banco (Monitoring, Sensor_Data)

Se nenhum CSV Kaggle for encontrado, retorna None e o pipeline
mantem o gerador sintetico como fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR


KAGGLE_DIR = DATA_DIR / "kaggle"

COLUMN_SYNONYMS = {
    "timestamp":   ["timestamp", "datetime", "date_time", "date", "time", "dt", "reading_time"],
    "pH":          ["ph", "p_h", "ph_value", "ph_level"],
    # TDS (ppm) vira EC (mS/cm) via divisao por 500 quando escolhido
    "EC":          ["ec", "conductivity", "electrical_conductivity", "ec_ms", "ec_mscm", "ec_value", "tds"],
    # em hidroponia a temperatura relevante e a da solucao (water_temp)
    "temperature": ["water_temp", "temperature", "temp", "temp_c", "temperature_c", "solution_temp"],
    "humidity":    ["dht_humidity", "humidity", "hum", "rh", "relative_humidity", "moisture"],
    "plant_id":    ["plant_id", "plant", "crop_id", "planta"],
    "growth":      ["growth_stage", "stage", "phase", "growth"],
    "yield":       ["yield", "harvest_weight", "weight", "production"],
    "is_default":  ["isdefault", "is_default", "default_flag"],
}

# Ordem de preferencia entre arquivos Kaggle (padrao case-insensitive)
# Prefere o RAW: dados brutos, sem interpolacao artificial, com timestamps
FILE_PREFERENCE = [
    "raw",                    # dados brutos (preferencia principal)
    "isdefaultinterpolate",   # limpo + com timestamp (fallback)
    "cleaned",
    "with_interpolation",
    "without_interpolation",
]

# Valores tipicos de setup/default do sistema IoT (quando nao ha coluna isDefault)
DEFAULT_SETUP_SIGNATURE = {
    "pH": 7.0,
    "TDS": 500.0,
    "water_temp": 20.0,
}

# Faixas fisicas plausiveis para hidroponia (validacao Quadro 1)
VALID_RANGES = {
    "pH":          (4.0, 8.5),   # extra faixa alem do ideal 5.5-6.5 para tolerar oscilacoes
    "EC":          (0.5, 4.0),   # mS/cm
    "temperature": (15.0, 35.0), # solucao aquosa em ambiente controlado
    "humidity":    (20.0, 100.0),
}

RESAMPLE_INTERVAL = "1min"  # granularidade fina: ~6600 bins, permite janelas longas


@dataclass
class MappingReport:
    file: Path
    original_rows: int
    original_cols: int
    dropped_nulls: int
    dropped_duplicates: int
    mapped_columns: dict[str, str]
    final_rows: int
    final_cols: list[str]

    def as_dict(self) -> dict:
        return {
            "arquivo":              str(self.file),
            "linhas_originais":     self.original_rows,
            "colunas_originais":    self.original_cols,
            "nulos_removidos":      self.dropped_nulls,
            "duplicatas_removidas": self.dropped_duplicates,
            "linhas_finais":        self.final_rows,
            "mapeamento":           self.mapped_columns,
            "colunas_finais":       self.final_cols,
        }


def _find_column(df_cols: list[str], target: str) -> str | None:
    lower_map = {c.lower().strip(): c for c in df_cols}
    for syn in COLUMN_SYNONYMS[target]:
        if syn in lower_map:
            return lower_map[syn]
    # tenta busca aproximada (substring)
    for lower, orig in lower_map.items():
        for syn in COLUMN_SYNONYMS[target]:
            if syn in lower:
                return orig
    return None


def _synthesize_timestamp(n: int, freq_min: int = 15) -> pd.Series:
    start = datetime(2026, 2, 1, 0, 0, 0)
    return pd.Series([start + timedelta(minutes=i * freq_min) for i in range(n)])


def _pick_best_file() -> Path | None:
    csvs = list(KAGGLE_DIR.glob("*.csv"))
    if not csvs:
        return None
    # ordena por preferencia baseada no nome
    def rank(p: Path) -> int:
        name = p.name.lower()
        for i, pat in enumerate(FILE_PREFERENCE):
            if pat in name:
                return i
        return len(FILE_PREFERENCE)
    csvs.sort(key=rank)
    return csvs[0]


def load_kaggle_csv(path: Path | None = None) -> tuple[pd.DataFrame, MappingReport] | None:
    """
    Lê um CSV do Kaggle e retorna DataFrame padronizado com colunas
    ['timestamp', 'pH', 'EC', 'temperature', 'humidity'] mais campos opcionais.

    Args:
        path: caminho especifico ou None para escolher o melhor CSV em data/kaggle/.
    """
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)

    if path is None:
        path = _pick_best_file()
        if path is None:
            return None

    df_raw = pd.read_csv(path, low_memory=False)
    n0, c0 = df_raw.shape
    mapping: dict[str, str] = {}

    # detecta colunas
    for target in ["timestamp", "pH", "EC", "temperature", "humidity",
                   "plant_id", "growth", "yield", "is_default"]:
        found = _find_column(list(df_raw.columns), target)
        if found is not None:
            mapping[target] = found

    df = pd.DataFrame()
    if "timestamp" in mapping:
        df["timestamp"] = pd.to_datetime(df_raw[mapping["timestamp"]], errors="coerce")
    else:
        df["timestamp"] = _synthesize_timestamp(len(df_raw))

    for col in ["pH", "EC", "temperature", "humidity"]:
        if col in mapping:
            df[col] = pd.to_numeric(df_raw[mapping[col]], errors="coerce")
        else:
            df[col] = np.nan

    # TDS (ppm) -> EC (mS/cm) usando fator 500 (aproximacao padrao)
    if "EC" in mapping and mapping["EC"].lower() == "tds":
        df["EC"] = df["EC"] / 500.0

    if "plant_id" in mapping:
        df["plant_id_orig"] = df_raw[mapping["plant_id"]].astype(str)
    if "growth" in mapping:
        df["growth_stage"] = df_raw[mapping["growth"]].astype(str)
    if "yield" in mapping:
        df["yield_kg"] = pd.to_numeric(df_raw[mapping["yield"]], errors="coerce")

    # remove linhas marcadas como valores default (setup do sistema)
    dropped_default = 0
    if "is_default" in mapping:
        default_flag = pd.to_numeric(df_raw[mapping["is_default"]], errors="coerce").fillna(0)
        keep_mask = default_flag == 0
        dropped_default = int((~keep_mask).sum())
        df = df.loc[keep_mask.values].reset_index(drop=True)
    else:
        # heuristica: linhas com assinatura de default de setup (pH=7 + TDS=500 + temp=20)
        setup_mask = pd.Series([True] * len(df))
        if "pH" in df.columns:
            setup_mask &= (df["pH"] == DEFAULT_SETUP_SIGNATURE["pH"])
        # TDS foi convertido em EC via /500 -> valor 1.0
        if "EC" in df.columns:
            setup_mask &= (df["EC"].round(3) == 1.0)
        if "temperature" in df.columns:
            setup_mask &= (df["temperature"] == DEFAULT_SETUP_SIGNATURE["water_temp"])
        dropped_default = int(setup_mask.sum())
        df = df.loc[~setup_mask].reset_index(drop=True)

    n_dupes = int(df.duplicated().sum())
    df = df.drop_duplicates()

    n_nulls = int(df[["pH", "EC", "temperature", "humidity"]].isna().any(axis=1).sum())
    df = df.dropna(subset=["pH", "EC", "temperature", "humidity"]).reset_index(drop=True)

    if "timestamp" in mapping:
        df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # remove outliers fora das faixas fisicas plausiveis
    n_before_out = len(df)
    for col, (lo, hi) in VALID_RANGES.items():
        df = df.loc[(df[col] >= lo) & (df[col] <= hi)]
    n_outliers = n_before_out - len(df)
    df = df.reset_index(drop=True)

    # reamostragem para 15 min (media do intervalo)
    n_before_resample = len(df)
    if "timestamp" in mapping:
        df_r = (df.set_index("timestamp")[["pH", "EC", "temperature", "humidity"]]
                  .resample(RESAMPLE_INTERVAL)
                  .mean()
                  .dropna()
                  .reset_index())
        df = df_r

    report = MappingReport(
        file=path,
        original_rows=n0,
        original_cols=c0,
        dropped_nulls=n_nulls + dropped_default + n_outliers,
        dropped_duplicates=n_dupes,
        mapped_columns=mapping,
        final_rows=len(df),
        final_cols=list(df.columns),
    )
    # anexa metadados extras uteis para depuracao
    report.mapped_columns = {**mapping,
                             "_resampled_15min": n_before_resample,
                             "_outliers_removidos": n_outliers}
    return df, report


def to_ml_csv(df: pd.DataFrame, dest: Path) -> None:
    """Salva o CSV padronizado que sera lido por data_loader.py."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    df[["timestamp", "pH", "EC", "temperature", "humidity"]].to_csv(dest, index=False)


if __name__ == "__main__":
    r = load_kaggle_csv()
    if r is None:
        print(f"Nenhum CSV encontrado em {KAGGLE_DIR}. Coloque o dataset do Kaggle la.")
    else:
        df, rep = r
        print("Resumo do tratamento (Quadro 1):")
        for k, v in rep.as_dict().items():
            print(f"  {k:25s} {v}")
        print("\nPrimeiras linhas:")
        print(df.head())
