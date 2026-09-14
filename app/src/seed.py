"""
ETL: aplica o schema no SQLite e carrega o dataset.

Fonte:
    - Se houver CSV em `data/kaggle/`, usa dado real (com limpeza/mapeamento)
    - Caso contrario, usa o gerador sintetico reprodutivel

Uso:
    python -m src.seed
"""
from __future__ import annotations

import pandas as pd

from src.database import connect, apply_schema, insert_dataframe
from src.kaggle_loader import load_kaggle_csv
from src.synthetic_dataset import generate_all, save_to_csv


def seed(use_kaggle: bool = True) -> dict[str, int]:
    apply_schema()

    real_ts = None
    if use_kaggle:
        result = load_kaggle_csv()
        if result is not None:
            df, rep = result
            print("Fonte: Kaggle CSV")
            print("Quadro 1 - Resumo do tratamento dos dados")
            print("-" * 60)
            for k, v in rep.as_dict().items():
                print(f"  {k:25s} {v}")
            real_ts = df
        else:
            print("Nenhum CSV em data/kaggle/ - usando gerador sintetico.")

    tables = generate_all(real_ts=real_ts)
    save_to_csv(tables)

    counts: dict[str, int] = {}
    with connect() as conn:
        # dimensões
        insert_dataframe("farm_status", tables["farm_status"], conn)
        insert_dataframe("ambient",     tables["ambient"], conn)
        insert_dataframe("users",       tables["users"], conn)
        insert_dataframe("sensor",      tables["sensor"], conn)
        insert_dataframe("nutrient",    tables["nutrient"], conn)
        insert_dataframe("plants",      tables["plants"], conn)

        # monitoring precisa de FKs válidas; ids são sequenciais
        insert_dataframe("monitoring", tables["monitoring"], conn)

        # SENSOR_DATA e MONITORING_NUTRIENT usam índice do dataframe -> converte para id (idx+1)
        sd = tables["sensor_data"].copy()
        sd["id_monitoring"] = sd.pop("id_monitoring_idx") + 1
        sd = sd[["id_monitoring", "id_sensor", "value"]]
        insert_dataframe("sensor_data", sd, conn)

        mn = tables["monitoring_nutrient"].copy()
        mn["id_monitoring"] = mn.pop("id_monitoring_idx") + 1
        mn["id_nutrient"] = mn.pop("id_nutrient_idx") + 1
        mn = mn[["id_monitoring", "id_nutrient"]]
        insert_dataframe("monitoring_nutrient", mn, conn)

        insert_dataframe("farm", tables["farm"], conn)

        # coletar contagens finais
        for tbl in [
            "farm_status", "plants", "ambient", "users", "sensor",
            "monitoring", "sensor_data", "nutrient", "monitoring_nutrient", "farm",
        ]:
            counts[tbl] = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    return counts


if __name__ == "__main__":
    counts = seed()
    print("\nQuadro 2 - Entidades do modelo relacional e volume de registros")
    print("-" * 60)
    for tbl, n in counts.items():
        print(f"  {tbl:22s} {n:>6d} registros")
