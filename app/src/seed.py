"""
Inicializa o banco: aplica o schema e carrega apenas os catalogos de
referencia (status de cultivo, tipos de sensor, nutrientes).

Nenhuma medicao e gerada aqui. O historico do sistema vem dos sensores,
pela API do ESP32 (`src.api_ingest`); plantas, ambientes, responsaveis e
colheitas sao cadastrados pelas telas RF01, RF02 e RF05.

Uso:
    python -m src.seed
"""
from __future__ import annotations

import pandas as pd

from src.api_ingest import aplicar_schema_api
from src.config import API_SENSOR_MAP
from src.database import apply_schema, connect, insert_dataframe

#: Entidades que nascem vazias: so recebem linha de sensor real ou de cadastro.
TABELAS_TRANSACIONAIS = [
    "plants", "ambient", "users", "monitoring",
    "sensor_data", "monitoring_nutrient", "farm",
]

#: Ciclo de vida de uma planta (dominio do documento, secao 3.2).
FARM_STATUS = ["Plantada", "Germinacao", "Crescimento", "Producao", "Colhida"]

#: Macro e micronutrientes acompanhados no cultivo.
NUTRIENTES = [
    "Nitrogenio (N)", "Fosforo (P)", "Potassio (K)",
    "Calcio (Ca)", "Magnesio (Mg)", "Enxofre (S)",
    "Ferro (Fe)", "Manganes (Mn)", "Zinco (Zn)",
    "Cobre (Cu)", "Boro (B)", "Molibdenio (Mo)",
]

#: Catalogo de sensores: exatamente as grandezas que o ESP32 publica.
DESCRICAO_SENSOR = {
    "pH":                   "Sensor eletroquimico de pH com compensacao de temperatura",
    "EC":                   "Sensor de condutividade eletrica da solucao nutritiva",
    "Temperatura":          "Sonda de temperatura da solucao nutritiva (DS18B20)",
    "Temperatura Ambiente": "Sensor de temperatura do ar da estufa (DHT)",
}


def tabela_sensores() -> pd.DataFrame:
    """Catalogo derivado de `config.API_SENSOR_MAP`: uma linha por grandeza medida."""
    linhas = [
        {"sensor_type": tipo, "unit": unidade,
         "description": DESCRICAO_SENSOR.get(tipo, f"Sensor {tipo}")}
        for tipo, unidade, _feature in API_SENSOR_MAP.values()
    ]
    return pd.DataFrame(linhas)


def seed() -> dict[str, int]:
    """
    Recria o banco vazio e popula os catalogos de referencia.
    Retorna a contagem de registros por entidade.
    """
    apply_schema()          # DROP/CREATE das 10 entidades do Apendice A
    aplicar_schema_api()    # tabelas de controle da ingestao

    counts: dict[str, int] = {}
    with connect() as conn:
        insert_dataframe("farm_status", pd.DataFrame({"status_name": FARM_STATUS}), conn)
        insert_dataframe("nutrient", pd.DataFrame({"name": NUTRIENTES}), conn)
        insert_dataframe("sensor", tabela_sensores(), conn)

        for tbl in [
            "farm_status", "plants", "ambient", "users", "sensor",
            "monitoring", "sensor_data", "nutrient", "monitoring_nutrient", "farm",
        ]:
            counts[tbl] = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    return counts


if __name__ == "__main__":
    counts = seed()
    print("Banco inicializado - catalogos de referencia carregados")
    print("-" * 62)
    for tbl, n in counts.items():
        marca = "catalogo" if n else "vazia"
        print(f"  {tbl:22s} {n:>4d} registros  ({marca})")
    print(
        "\nAs medicoes vem dos sensores. Importe com:\n"
        "    python -m src.api_ingest"
    )
