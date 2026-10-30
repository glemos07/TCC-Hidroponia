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

# O catalogo de sensores vive em API_SENSOR_MAP (abaixo): e a lista de
# grandezas que o hardware realmente mede, e dela saem tanto as linhas da
# tabela SENSOR quanto as features dos modelos.

FORECAST_HORIZON_MIN = 60         # extremo superior das faixas 15-60 min do doc
SAMPLING_INTERVAL_MIN = 1         # passo da serie reamostrada
FORECAST_HORIZON_STEPS = FORECAST_HORIZON_MIN // SAMPLING_INTERVAL_MIN  # = 60 passos
LSTM_WINDOW = 240                 # 4 horas de contexto (240 * 1 min)


# ---------------------------------------------------------------------------
# Integracao com a API de sensores (servidor FastAPI do ESP32)
# ---------------------------------------------------------------------------
import os as _os

#: URL base do servidor FastAPI (servidor/servidor_api.py).
#: Sobrescreva com a variavel de ambiente HIDRO_API_URL.
API_BASE_URL = _os.getenv("HIDRO_API_URL", "http://localhost:8000")

#: Fazenda padrao consultada na API (vazio = todas).
API_ID_FAZENDA = _os.getenv("HIDRO_API_FAZENDA", "")

#: Timeout (s) e tamanho de pagina das requisicoes HTTP.
API_TIMEOUT = float(_os.getenv("HIDRO_API_TIMEOUT", "10"))
API_PAGE_SIZE = 1000          # limite maximo aceito por /v1/leituras

#: DDL complementar: tabelas de rastreio da ingestao da API.
SCHEMA_API_SQL = BASE_DIR / "db" / "schema_api.sql"

# ---------------------------------------------------------------------------
# Conjuntos de features por fonte de dados
#
# O dataset Kaggle traz umidade do ar; a API do ESP32 nao tem sensor de
# umidade, mas tem DUAS temperaturas (solucao e ambiente). Por isso cada
# fonte tem seu proprio vetor de entrada e seus proprios artefatos treinados
# - os modelos do Quadro 6 (fonte "kaggle") continuam validos e inalterados.
# ---------------------------------------------------------------------------
FEATURE_SETS = {
    "kaggle": ["pH", "EC", "temperature", "humidity"],
    "api":    ["pH", "EC", "temperature", "air_temp"],
}

#: Sufixo dos arquivos de modelo por fonte (kaggle mantem os nomes originais).
ARTIFACT_SUFFIX = {"kaggle": "", "api": "_api"}

#: Campo da API -> (sensor_type no banco, unidade, feature do modelo)
#: Em hidroponia a temperatura relevante e a da solucao nutritiva, por isso
#: `temp_agua` assume o papel de "Temperatura" (mesma convencao do kaggle_loader).
API_SENSOR_MAP = {
    "ph":            ("pH",                   "escala", "pH"),
    "ec":            ("EC",                   "mS/cm",  "EC"),
    "temp_agua":     ("Temperatura",          "C",      "temperature"),
    "temp_ambiente": ("Temperatura Ambiente", "C",      "air_temp"),
}

#: Faixas fisicas plausiveis por campo da API. Mesma validacao do Quadro 1
#: aplicada ao dataset Kaggle (VALID_RANGES em src/kaggle_loader.py): medidas
#: fora da faixa sao falha de sensor, nao variacao do cultivo, e entrariam como
#: ruido no treino. A faixa e mais larga que a ideal (PH_MIN..PH_MAX) de
#: proposito - o objetivo e cortar o impossivel, nao o indesejado.
API_FAIXAS_VALIDAS = {
    "ph":            (4.0, 8.5),
    "ec":            (0.5, 4.0),    # mS/cm
    "temp_agua":     (15.0, 35.0),  # solucao em ambiente controlado
    "temp_ambiente": (0.0, 50.0),   # ar da estufa
}

#: Lacuna maxima (em multiplos do intervalo mediano) tolerada dentro de uma
#: janela do LSTM. Acima disso a janela cruza uma interrupcao da coleta e nao
#: representa dinamica continua.
MAX_GAP_FACTOR = 5.0

#: sensor_type no banco -> nome da feature usada pelos modelos.
SENSOR_TO_FEATURE = {
    "pH":                   "pH",
    "EC":                   "EC",
    "Temperatura":          "temperature",
    "Umidade":              "humidity",
    "Temperatura Ambiente": "air_temp",
}


def features_de(fonte: str = "kaggle") -> list[str]:
    """Vetor de entrada dos modelos para a fonte informada."""
    try:
        return list(FEATURE_SETS[fonte])
    except KeyError:
        raise ValueError(f"Fonte desconhecida: {fonte!r}. Use {sorted(FEATURE_SETS)}.") from None
