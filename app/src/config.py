"""Configurações centrais do projeto TCC - Hidroponia Inteligente."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "db"
MODELS_DIR = BASE_DIR / "models" / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"

for d in (DATA_DIR, DB_DIR, MODELS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

DB_PATH = DB_DIR / "hidroponia.sqlite3"
DB_URL = f"sqlite:///{DB_PATH.as_posix()}"

SCHEMA_SQL = BASE_DIR / "db" / "schema.sql"

RANDOM_SEED = 42

PH_MIN, PH_MAX = 5.5, 6.5
EC_MIN, EC_MAX = 1.0, 3.5

SENSOR_TYPES = {
    "pH": ("pH", "escala"),
    "EC": ("Condutividade", "mS/cm"),
    "TEMP": ("Temperatura", "°C"),
    "HUM": ("Umidade", "%"),
    "NUT": ("Nutrientes", "ppm"),
}

FORECAST_HORIZON_MIN = 60         # extremo superior das faixas 15-60 min do doc
SAMPLING_INTERVAL_MIN = 1         # dados Kaggle reamostrados a 1 min
FORECAST_HORIZON_STEPS = FORECAST_HORIZON_MIN // SAMPLING_INTERVAL_MIN  # = 60 passos
LSTM_WINDOW = 240                 # 4 horas de contexto (240 * 1 min)
ML_EXTRA_DAYS = 45                # dias extras de serie ML sintetica (fallback)
