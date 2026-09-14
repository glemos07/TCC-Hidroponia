"""
Casos de teste CT01 a CT06 conforme secao 5.9 e Quadro 5 do documento.
"""
import sqlite3

import pytest

from src.database import connect


def test_ct01_cadastrar_planta_e_persistir(db):
    """CT01 - Cadastrar uma nova planta e verificar a persistencia do registro."""
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO plants (name_plant, diversity, dt_planting, status_id) "
            "VALUES ('Rucula Nova', 'Rucula', '2026-02-15', 1)"
        )
        new_id = cur.lastrowid
        row = conn.execute("SELECT name_plant FROM plants WHERE id=?", (new_id,)).fetchone()
    assert row is not None
    assert row["name_plant"] == "Rucula Nova"


def test_ct02_cadastrar_ambiente_e_persistir(db):
    """CT02 - Cadastrar um ambiente de cultivo e confirmar a persistencia."""
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO ambient (nm_ambient, tp_ambient, descriptions) "
            "VALUES ('Ambiente-B', 'Aeroponia', 'Setor B')"
        )
        new_id = cur.lastrowid
        row = conn.execute("SELECT nm_ambient, tp_ambient FROM ambient WHERE id=?",
                           (new_id,)).fetchone()
    assert row["nm_ambient"] == "Ambiente-B"
    assert row["tp_ambient"] == "Aeroponia"


def test_ct03_inserir_monitoramento_vinculado(db):
    """CT03 - Inserir monitoramento vinculado a planta e ambiente."""
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO monitoring (dt_register, id_plant, id_ambient, id_user) "
            "VALUES ('2026-02-20 10:00:00', 1, 1, 1)"
        )
        mid = cur.lastrowid
        conn.execute(
            "INSERT INTO sensor_data (id_monitoring, id_sensor, value) VALUES (?, 1, 6.1)",
            (mid,),
        )
        n = conn.execute("SELECT COUNT(*) AS c FROM sensor_data WHERE id_monitoring=?",
                         (mid,)).fetchone()["c"]
    assert n == 1


def test_ct04_consultar_historico_planta(db):
    """CT04 - Consultar o historico de medicoes de uma planta especifica."""
    with connect() as conn:
        for i in range(5):
            cur = conn.execute(
                "INSERT INTO monitoring (dt_register, id_plant, id_ambient, id_user) "
                "VALUES (?, 1, 1, 1)",
                (f"2026-02-2{i} 08:00:00",),
            )
            conn.execute(
                "INSERT INTO sensor_data (id_monitoring, id_sensor, value) VALUES (?, 1, ?)",
                (cur.lastrowid, 5.8 + i * 0.05),
            )
        rows = conn.execute("""
            SELECT sd.value FROM sensor_data sd
            JOIN monitoring m ON m.id = sd.id_monitoring
            WHERE m.id_plant = 1
        """).fetchall()
    assert len(rows) == 5


def test_ct05_registrar_producao_associada(db):
    """CT05 - Registrar producao final e verificar associacao correta com planta."""
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO farm (dt_harvest, weight_harvest, time_growth, id_plant, id_ambient) "
            "VALUES ('2026-03-15', 12.5, 38.0, 1, 1)"
        )
        fid = cur.lastrowid
        row = conn.execute(
            "SELECT id_plant, weight_harvest FROM farm WHERE id=?", (fid,)
        ).fetchone()
    assert row["id_plant"] == 1
    assert row["weight_harvest"] == pytest.approx(12.5)


def test_ct06_impede_insercao_com_fk_inexistente(db):
    """CT06 - Impedir a insercao de monitoramento com chave estrangeira inexistente."""
    with pytest.raises(sqlite3.IntegrityError):
        with connect() as conn:
            conn.execute(
                "INSERT INTO monitoring (dt_register, id_plant, id_ambient, id_user) "
                "VALUES ('2026-02-20 10:00:00', 9999, 1, 1)"
            )
