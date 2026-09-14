"""
Configuracao dos testes: cria um banco em arquivo temporario e aplica o schema.
Isolado do banco de producao para nao poluir dados reais.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# garante que `src` seja importavel a partir da raiz do app/
BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Banco SQLite isolado para cada teste."""
    from src import config

    test_db = tmp_path / "test.sqlite3"
    monkeypatch.setattr(config, "DB_PATH", test_db)

    from src.database import apply_schema, connect
    apply_schema()

    # seed minimo: 1 status, 1 planta, 1 ambiente, 1 sensor, 1 user
    with connect() as conn:
        conn.execute("INSERT INTO farm_status (status_name) VALUES ('Crescimento')")
        conn.execute(
            "INSERT INTO plants (name_plant, diversity, dt_planting, status_id) "
            "VALUES ('Alface Teste', 'Crespa', '2026-02-01', 1)"
        )
        conn.execute(
            "INSERT INTO ambient (nm_ambient, tp_ambient, descriptions) "
            "VALUES ('Ambiente-A', 'Estufa NFT', 'Teste')"
        )
        conn.execute(
            "INSERT INTO users (name, email, role) "
            "VALUES ('Tester', 'teste@unip.br', 'Operador')"
        )
        conn.execute(
            "INSERT INTO sensor (sensor_type, unit, description) "
            "VALUES ('pH', 'escala', 'Sensor de pH')"
        )
    return test_db
