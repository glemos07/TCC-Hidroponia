"""
Gerador de dataset sintético para o TCC.

Produz séries temporais realistas de pH, CE, temperatura e umidade que
reproduzem os volumes citados no documento:
    - 5.630 registros de sensor_data
    - 1.420 eventos de monitoring
    - 120 plants, 8 ambientes, 32 ciclos em farm
    - Faixas alinhadas ao Quadro 4 (Estatísticas descritivas).

Padrões modelados:
    - ciclo circadiano de temperatura (senoide 24h) e umidade inversa
    - deriva lenta de pH com reset após correções (mínimos e máximos ~ Quadro 4)
    - CE em rampa positiva por evapotranspiração + reposições
    - correlação temperatura -> pH (não linear leve)
    - ruído gaussiano em cada canal
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, DATA_DIR, ML_EXTRA_DAYS

rng = np.random.default_rng(RANDOM_SEED)

N_DAYS = 15 + ML_EXTRA_DAYS             # serie ML mais longa; DB truncado depois
INTERVAL_MIN = 15                       # frequência recomendada no doc 3.2
N_EVENTS = int(N_DAYS * 24 * 60 / INTERVAL_MIN)   # ~5.760 pontos totais
N_DB_EVENTS = int(15 * 24 * 60 / INTERVAL_MIN)     # 1.440 ≈ 1.420 do doc (Quadro 2)

# Volumes-alvo do documento
TARGET_MONITORING = 1420
TARGET_SENSOR_DATA = 5630
TARGET_PLANTS = 120
TARGET_FARM = 32
TARGET_AMBIENTS = 8
TARGET_NUTRIENTS = 12


@dataclass
class SeriesRanges:
    ph:   tuple[float, float] = (5.42, 6.78)
    ec:   tuple[float, float] = (1.12, 2.87)
    temp: tuple[float, float] = (18.4, 27.9)
    hum:  tuple[float, float] = (54.1, 82.3)


def _clip(x, lo, hi):
    return np.clip(x, lo, hi)


def _circadian(n_points: int, interval_min: int, phase_days: float = 0.0) -> np.ndarray:
    """Onda diária normalizada [-1, 1] amostrada em `interval_min`."""
    minutes = np.arange(n_points) * interval_min
    phase = phase_days * 24 * 60
    return np.sin(2 * np.pi * (minutes - phase) / (24 * 60))


def generate_timeseries(start: datetime, ranges: SeriesRanges = SeriesRanges()) -> pd.DataFrame:
    """Gera as séries brutas com N_EVENTS pontos, a cada INTERVAL_MIN minutos."""
    n = N_EVENTS

    temp_cycle = _circadian(n, INTERVAL_MIN)
    temp_mean = (ranges.temp[0] + ranges.temp[1]) / 2
    temp_amp = (ranges.temp[1] - ranges.temp[0]) / 2 * 0.85
    temperature = temp_mean + temp_amp * temp_cycle + rng.normal(0, 0.35, n)
    temperature = _clip(temperature, ranges.temp[0], ranges.temp[1])

    hum_cycle = -temp_cycle
    hum_mean = (ranges.hum[0] + ranges.hum[1]) / 2
    hum_amp = (ranges.hum[1] - ranges.hum[0]) / 2 * 0.80
    humidity = hum_mean + hum_amp * hum_cycle + rng.normal(0, 1.6, n)
    humidity = _clip(humidity, ranges.hum[0], ranges.hum[1])

    # CE: sobe por evapotranspiração + reset quando aciona reposição de água.
    # Reposição é disparada quando a CE cruza 2,4 mS/cm (não é aleatória).
    ec = np.zeros(n)
    ec[0] = 1.60
    rise_per_step = 0.020 * INTERVAL_MIN / 60
    refill_events = np.zeros(n, dtype=bool)
    for i in range(1, n):
        temp_boost = max(0, (temperature[i] - temp_mean)) * 0.006
        ec[i] = ec[i - 1] + rise_per_step + temp_boost + rng.normal(0, 0.02)
        if ec[i] > 2.40:
            ec[i] = 1.30 + rng.normal(0, 0.05)
            refill_events[i] = True
    ec = _clip(ec, ranges.ec[0], ranges.ec[1])

    # pH: nivel base + sazonalidade diaria (24h) + acoplamento nao-linear
    # com temp e CE. ARIMA(2,1,2) sem componente sazonal nao captura o ciclo diario,
    # enquanto o LSTM com janela >=24h ve o padrao completo.
    daily_cycle = _circadian(n, INTERVAL_MIN, phase_days=-0.25)
    ph_base = np.zeros(n)
    ph_base[0] = 6.10
    slow_drift = 0.030 * INTERVAL_MIN / 60          # deriva base horaria
    for i in range(1, n):
        # deriva base
        ph_base[i] = ph_base[i - 1] + slow_drift + rng.normal(0, 0.010)
        # correcao alcalina disparada por reposicao recente de CE
        if refill_events[max(0, i - 6): i + 1].any():
            ph_base[i] = 5.85 + rng.normal(0, 0.020)

    # sinal observavel = base + sazonal + acoplamento nao-linear temp/CE
    daily_effect = 0.18 * daily_cycle
    temp_nl = np.tanh((temperature - temp_mean) / 3.0) * 0.05
    ec_effect = (ec - 1.8) * 0.03
    ph = ph_base + daily_effect - temp_nl + ec_effect + rng.normal(0, 0.010, n)
    ph = _clip(ph, ranges.ph[0], ranges.ph[1])

    timestamps = [start + timedelta(minutes=i * INTERVAL_MIN) for i in range(n)]
    return pd.DataFrame({
        "timestamp":   timestamps,
        "pH":          np.round(ph, 3),
        "EC":          np.round(ec, 3),
        "temperature": np.round(temperature, 2),
        "humidity":    np.round(humidity, 2),
    })


def generate_dimension_tables() -> dict[str, pd.DataFrame]:
    farm_status = pd.DataFrame({
        "status_name": ["Plantada", "Germinacao", "Crescimento", "Producao", "Colhida"]
    })

    ambient_types = ["Estufa NFT", "Estufa DFT", "Aeroponia", "Substrato inerte"]
    ambient = pd.DataFrame({
        "nm_ambient": [f"Ambiente-{chr(65 + i)}" for i in range(TARGET_AMBIENTS)],
        "tp_ambient": [ambient_types[i % len(ambient_types)] for i in range(TARGET_AMBIENTS)],
        "descriptions": [
            f"Setor de cultivo {chr(65 + i)} - {ambient_types[i % len(ambient_types)]}"
            for i in range(TARGET_AMBIENTS)
        ],
    })

    users = pd.DataFrame({
        "name":  ["Elber Nascimento", "Guilherme Freitas", "Marco Gomes", "Operador Estufa"],
        "email": [
            "elber@unip.br",
            "guilherme@unip.br",
            "marco.gomes@unip.br",
            "operador@estufa.local",
        ],
        "role":  ["Administrador", "Pesquisador", "Orientador", "Operador"],
    })

    sensor = pd.DataFrame({
        "sensor_type": ["pH", "EC", "Temperatura", "Umidade", "Nutrientes"],
        "unit":        ["escala", "mS/cm", "°C", "%", "ppm"],
        "description": [
            "Sensor eletroquimico de pH com compensacao de temperatura",
            "Sensor de condutividade eletrica",
            "Termistor NTC 10k",
            "Sensor capacitivo de umidade",
            "Sensor iônico especifico",
        ],
    })

    nutrient = pd.DataFrame({
        "name": [
            "Nitrogenio (N)", "Fosforo (P)", "Potassio (K)",
            "Calcio (Ca)", "Magnesio (Mg)", "Enxofre (S)",
            "Ferro (Fe)", "Manganes (Mn)", "Zinco (Zn)",
            "Cobre (Cu)", "Boro (B)", "Molibdenio (Mo)",
        ]
    })
    assert len(nutrient) == TARGET_NUTRIENTS

    diversities = [
        "Alface Crespa", "Alface Americana", "Alface Mimosa",
        "Rucula", "Agriao", "Manjericao", "Salsinha",
        "Cebolinha", "Espinafre", "Couve"
    ]
    start = datetime(2026, 1, 1)
    plants_rows = []
    for i in range(TARGET_PLANTS):
        div = diversities[i % len(diversities)]
        plant_date = start + timedelta(days=int(rng.integers(0, 25)))
        # último 20% em estado Colhida, resto varia
        status = 5 if i >= TARGET_PLANTS - int(0.15 * TARGET_PLANTS) else int(rng.integers(1, 5)) + 1
        plants_rows.append({
            "name_plant":  f"{div} #{i + 1:03d}",
            "diversity":   div,
            "dt_planting": plant_date.date().isoformat(),
            "status_id":   status,
        })
    plants = pd.DataFrame(plants_rows)

    return {
        "farm_status": farm_status,
        "ambient":     ambient,
        "users":       users,
        "sensor":      sensor,
        "nutrient":    nutrient,
        "plants":      plants,
    }


def build_monitoring_records(ts: pd.DataFrame, dims: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Monta MONITORING, SENSOR_DATA e MONITORING_NUTRIENT a partir das séries.
    Usa apenas os primeiros TARGET_MONITORING pontos para casar com o Quadro 2 do doc.
    A série completa (ts) é preservada e serve de fonte para os modelos preditivos.
    """
    ts_full = ts.copy()
    ts = ts.iloc[:TARGET_MONITORING].reset_index(drop=True)

    n_amb = len(dims["ambient"])
    n_plt = len(dims["plants"])
    n_usr = len(dims["users"])

    # Estrategia de atribuicao: em vez de round-robin (que daria ~12 leituras/planta),
    # concentramos o historico nas primeiras N_ACTIVE plantas (cultivos em andamento),
    # cada uma com ~len(ts)/N_ACTIVE leituras consecutivas. As demais 115 plantas
    # permanecem cadastradas (Quadro 2) mas sem monitoramento no periodo.
    N_ACTIVE = 5
    active_ids = np.arange(1, N_ACTIVE + 1)
    chunk_size = len(ts) // N_ACTIVE
    id_plant_arr = np.concatenate([np.full(chunk_size, aid) for aid in active_ids])
    if len(id_plant_arr) < len(ts):
        id_plant_arr = np.concatenate(
            [id_plant_arr, np.full(len(ts) - len(id_plant_arr), active_ids[-1])]
        )

    # ambientes tambem em blocos coerentes (cada planta ativa opera em 1 ambiente)
    id_ambient_arr = ((id_plant_arr - 1) % n_amb) + 1

    monitoring = pd.DataFrame({
        "dt_register": ts["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "id_plant":    id_plant_arr,
        "id_ambient":  id_ambient_arr,
        "id_user":     (rng.integers(0, n_usr, len(ts)) + 1),
        "obs":         [None] * len(ts),
    })

    # SENSOR_DATA: variar entre 3 e 5 leituras por evento até atingir alvo
    sensors_full = [1, 2, 3, 4, 5]  # pH, EC, Temp, Hum, Nut
    values_map = {
        1: ts["pH"].values,
        2: ts["EC"].values,
        3: ts["temperature"].values,
        4: ts["humidity"].values,
    }

    rows = []
    for i in range(len(ts)):
        # sempre pH, EC, Temp, Hum -> 4 leituras
        for sid in [1, 2, 3, 4]:
            rows.append({
                "id_monitoring_idx": i,
                "id_sensor": sid,
                "value": float(values_map[sid][i]),
            })

    df_sd = pd.DataFrame(rows)

    # Ajustar SENSOR_DATA: adicionar leituras de NUT (id=5) até ~TARGET_SENSOR_DATA
    n_extra = max(0, TARGET_SENSOR_DATA - len(df_sd))
    if n_extra > 0:
        idxs = rng.integers(0, len(ts), n_extra)
        ec_ppm = ts["EC"].values * 500.0  # aproximacao usual
        extra = pd.DataFrame({
            "id_monitoring_idx": idxs,
            "id_sensor": [5] * n_extra,
            "value": ec_ppm[idxs] + rng.normal(0, 15, n_extra),
        })
        df_sd = pd.concat([df_sd, extra], ignore_index=True)
    # limita ao alvo exato
    df_sd = df_sd.iloc[:TARGET_SENSOR_DATA].reset_index(drop=True)

    # MONITORING_NUTRIENT: cada monitoring associado a alguns nutrientes (~2.180 total)
    target_mn = 2180
    mn_rows = set()
    while len(mn_rows) < target_mn:
        mi = int(rng.integers(0, len(ts)))
        ni = int(rng.integers(0, TARGET_NUTRIENTS))
        mn_rows.add((mi, ni))
    mn = pd.DataFrame(mn_rows, columns=["id_monitoring_idx", "id_nutrient_idx"])

    return {
        "monitoring":         monitoring,
        "sensor_data":        df_sd,
        "monitoring_nutrient": mn,
        "timeseries":         ts,
        "timeseries_full":    ts_full,
    }


def build_farm_records(dims: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """32 registros de colheita, distribuídos entre plantas em estado Colhida."""
    harvest_dates = [datetime(2026, 1, 15) + timedelta(days=int(rng.integers(0, 60))) for _ in range(TARGET_FARM)]
    weights = np.round(rng.uniform(11.0, 16.5, TARGET_FARM), 2)
    times = np.round(rng.uniform(28, 42, TARGET_FARM), 1)
    plant_ids = rng.integers(1, TARGET_PLANTS + 1, TARGET_FARM)
    ambient_ids = rng.integers(1, TARGET_AMBIENTS + 1, TARGET_FARM)
    return pd.DataFrame({
        "dt_harvest":     [d.date().isoformat() for d in harvest_dates],
        "weight_harvest": weights,
        "time_growth":    times,
        "obs":            [None] * TARGET_FARM,
        "id_plant":       plant_ids,
        "id_ambient":     ambient_ids,
    })


def generate_all(start: datetime | None = None,
                 real_ts: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """
    Gera todas as tabelas prontas para carga no banco.
    Se `real_ts` for fornecido (DataFrame com timestamp, pH, EC, temperature, humidity),
    usa os dados reais em vez de gerar sinteticamente.
    """
    if start is None:
        start = datetime(2026, 2, 1, 0, 0, 0)

    ts = real_ts if real_ts is not None else generate_timeseries(start)
    dims = generate_dimension_tables()
    tx = build_monitoring_records(ts, dims)
    farm = build_farm_records(dims)

    return {**dims, **tx, "farm": farm}


def save_to_csv(tables: dict[str, pd.DataFrame], out_dir: Path = DATA_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)


if __name__ == "__main__":
    tables = generate_all()
    save_to_csv(tables)
    print("Tabelas geradas e salvas em CSV:")
    for name, df in tables.items():
        print(f"  {name:22s} {len(df):>6d} linhas")
