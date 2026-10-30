"""
Testes da ingestao de dados vindos da API de sensores (ESP32).

Nao sobem o servidor FastAPI: exercitam a normalizacao do lote e a gravacao
no modelo relacional, que e onde mora a regra de negocio.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.api_client import normalizar, renomear_para_features, resumo_simulados
from src.api_ingest import gravar_lote, resumo_ingestao, ultima_leitura_importada
from src.database import connect


def _bruto(n: int = 3, id_planta: str = "alface-01") -> pd.DataFrame:
    """Lote como a API devolve: timestamps ISO com offset e campos em minusculo."""
    return pd.DataFrame([
        {
            "id": i + 1,
            "id_fazenda": "fazenda-01",
            "id_planta": id_planta,
            "nome_planta": "Alface Crespa",
            "timestamp": f"2026-10-30T08:{i:02d}:00-03:00",
            "temp_agua": 21.0 + i * 0.1,
            "temp_ambiente": 26.0 + i * 0.2,
            "ph": 6.0 + i * 0.01,
            "ec": 1.8 + i * 0.02,
            "sensores_simulados": None,
            "recebido_em": f"2026-10-30T08:{i:02d}:05-03:00",
        }
        for i in range(n)
    ])


def test_normalizar_tira_fuso_e_ordena():
    df = normalizar(_bruto(3).iloc[::-1])          # chega fora de ordem
    assert list(df["timestamp"]) == sorted(df["timestamp"])
    assert df["timestamp"].dt.tz is None           # datetime local, sem fuso
    assert df["timestamp"].iloc[0].hour == 8       # -03:00 preservado como horario local
    assert df["ph"].dtype.kind == "f"


def test_normalizar_lote_vazio_mantem_colunas():
    df = normalizar(pd.DataFrame())
    assert df.empty
    for col in ("timestamp", "ph", "ec", "temp_agua", "temp_ambiente"):
        assert col in df.columns


def test_renomeio_para_features():
    df = renomear_para_features(normalizar(_bruto(1)))
    assert {"pH", "EC", "temperature", "air_temp"} <= set(df.columns)


def test_resumo_simulados():
    assert resumo_simulados([None, "ph,ec", "ph"]) == "ph, ec"
    assert resumo_simulados([None, "False", float("nan")]) == "nenhum"


def test_gravar_lote_cria_entidades(db):
    rel = gravar_lote(normalizar(_bruto(3)))

    assert rel.recebidas == 3
    assert rel.novas == 3
    assert rel.medidas_inseridas == 12          # 3 leituras x 4 sensores
    assert rel.plantas_criadas                  # planta criada na 1a vez

    with connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM monitoring").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM sensor_data").fetchone()[0] == 12
        tipos = {r[0] for r in conn.execute("SELECT sensor_type FROM sensor")}
    assert {"pH", "EC", "Temperatura", "Temperatura Ambiente"} <= tipos


def test_gravar_lote_e_idempotente(db):
    lote = normalizar(_bruto(3))
    gravar_lote(lote)
    rel = gravar_lote(lote)                     # mesma carga de novo

    assert rel.novas == 0
    assert rel.ja_importadas == 3
    with connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM monitoring").fetchone()[0] == 3


def test_leitura_com_sensores_nulos_e_descartada(db):
    bruto = _bruto(2)
    bruto.loc[1, ["ph", "ec", "temp_agua", "temp_ambiente"]] = None
    rel = gravar_lote(normalizar(bruto))

    assert rel.novas == 1                       # a leitura sem nenhum sensor nao vira monitoramento
    with connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM monitoring").fetchone()[0] == 1


def test_marcador_incremental_acompanha_ultima_leitura(db):
    assert ultima_leitura_importada() is None
    gravar_lote(normalizar(_bruto(3)))

    marcador = ultima_leitura_importada()
    assert marcador is not None
    assert marcador.strftime("%H:%M") == "08:02"


def test_duas_plantas_viram_duas_linhas_em_plants(db):
    lote = pd.concat([_bruto(2, "alface-01"), _bruto(2, "rucula-02")], ignore_index=True)
    lote["id"] = range(1, len(lote) + 1)

    gravar_lote(normalizar(lote))
    resumo = resumo_ingestao()

    assert len(resumo) == 2
    assert set(resumo["id_planta"]) == {"alface-01", "rucula-02"}
    assert resumo["id_plant"].nunique() == 2


def test_ids_diferentes_com_mesmo_timestamp_nao_duplicam(db):
    """
    Builds antigas do servidor aceitaram a mesma leitura duas vezes com ids
    distintos. O par (planta, instante) tem de bastar para reconhece-la.
    """
    lote = pd.concat([_bruto(3), _bruto(3)], ignore_index=True)
    lote["id"] = range(1, 7)                    # ids novos, timestamps repetidos

    rel = gravar_lote(normalizar(lote))
    assert rel.novas == 3

    with connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM monitoring").fetchone()[0] == 3
        # todo monitoramento gravado continua rastreado (nenhum orfao)
        orfaos = conn.execute(
            "SELECT COUNT(*) FROM monitoring m "
            "WHERE NOT EXISTS (SELECT 1 FROM api_leitura a WHERE a.id_monitoring = m.id)"
        ).fetchone()[0]
    assert orfaos == 0

    # reexecutar nao recria nada
    assert gravar_lote(normalizar(lote)).novas == 0


def test_medidas_fora_da_faixa_fisica_sao_descartadas(db):
    bruto = _bruto(3)
    bruto.loc[0, "ph"] = 13.1        # impossivel em hidroponia
    bruto.loc[1, "ec"] = 25.0        # 17x o normal
    rel = gravar_lote(normalizar(bruto))

    assert rel.novas == 3                        # as leituras entram...
    assert rel.medidas_inseridas == 10           # ...sem as 2 medidas invalidas
    assert rel.medidas_fora_faixa == {"ph": 1, "ec": 1}

    with connect() as conn:
        maximo = conn.execute(
            "SELECT MAX(sd.value) FROM sensor_data sd "
            "JOIN sensor s ON s.id = sd.id_sensor WHERE s.sensor_type = 'pH'"
        ).fetchone()[0]
    assert maximo < 8.5


def test_leitura_toda_fora_da_faixa_nao_vira_monitoramento(db):
    bruto = _bruto(1)
    bruto.loc[0, ["ph", "ec", "temp_agua", "temp_ambiente"]] = [13.1, 25.0, 99.0, 99.0]
    rel = gravar_lote(normalizar(bruto))

    assert rel.novas == 0
    assert rel.descartadas == 1


def test_janelas_que_cruzam_lacuna_sao_descartadas():
    """Uma parada de coleta no meio da janela invalida aquela amostra."""
    import numpy as np

    from src.model_lstm import build_supervised_delta

    n = 40
    ts = pd.date_range("2026-10-30 08:00", periods=n, freq="1min").to_numpy()
    valores = np.random.RandomState(0).rand(n, 4)
    alvo = valores[:, 0]

    sem_filtro, *_ = build_supervised_delta(valores, alvo, window=5, target_idx=0, horizon=3)

    ts_com_lacuna = ts.copy()
    ts_com_lacuna[20:] += np.timedelta64(6, "h")      # 6 h parado no meio
    com_filtro, *_ = build_supervised_delta(
        valores, alvo, window=5, target_idx=0, horizon=3,
        timestamps=ts_com_lacuna, max_gap_min=5.0,
    )

    assert len(com_filtro) < len(sem_filtro)
    assert len(com_filtro) == len(sem_filtro) - (5 + 3 - 1)


def test_janela_limitada_bate_com_o_fim_do_historico(db):
    """
    Pedir so as ultimas N leituras tem de devolver exatamente a cauda do
    historico completo - e a otimizacao que mantem a inferencia constante
    conforme o banco cresce.
    """
    from src.config import features_de
    from src.data_loader import load_plant_features, plantas_api

    bruto = _bruto(40)
    bruto["timestamp"] = [f"2026-10-30T08:{i:02d}:00-03:00" for i in range(40)]
    gravar_lote(normalizar(bruto))

    # a ingestao cria sua propria planta; o fixture ja tinha outra
    id_plant = int(plantas_api()["id_plant"].iloc[0])
    feats = features_de("api")
    completo = load_plant_features(id_plant, feats)
    cauda = load_plant_features(id_plant, feats, limite=10)

    assert len(completo) == 40
    assert len(cauda) == 10
    pd.testing.assert_frame_equal(
        completo.tail(10).reset_index(drop=True), cauda.reset_index(drop=True)
    )


def test_backtest_alinha_previsto_com_o_real_correto(db, monkeypatch):
    """
    O backtest tem de comparar cada previsao com a observacao de exatamente um
    horizonte depois da janela - um desalinhamento aqui inflaria ou mascararia
    o erro sem que nada quebrasse.
    """
    import numpy as np

    from src import predict as mod

    # janela e horizonte agora vem da config, por fonte
    monkeypatch.setattr(mod, "passos_janela", lambda fonte: 3)
    monkeypatch.setattr(mod, "passos_horizonte", lambda fonte: 2)
    monkeypatch.setattr(mod, "API_RESAMPLE_MIN", None)   # sem reamostrar no teste

    n = 20
    bruto = _bruto(n)
    bruto["timestamp"] = [f"2026-10-30T08:{i:02d}:00-03:00" for i in range(n)]
    bruto["ph"] = [6.0 + i * 0.01 for i in range(n)]
    gravar_lote(normalizar(bruto))

    class ModeloFalso:
        """Devolve sempre delta zero: assim previsto == valor de referencia."""
        def predict(self, x, verbose=0):
            return np.zeros((len(x), 1))

    monkeypatch.setattr(mod, "_carregar",
                        lambda alvo, fonte: (ModeloFalso(), _ScalerIdentidade()))

    from src.data_loader import plantas_api
    id_plant = int(plantas_api()["id_plant"].iloc[0])

    bt = mod.backtest_recente(id_plant, "pH", "api", n_pontos=5)
    assert bt is not None and len(bt) == 5

    # com delta zero, previsto = valor da referencia, um horizonte-1 antes do real
    hist = mod.load_plant_history(id_plant, "api")
    for linha in bt.itertuples(index=False):
        idx_real = hist.index[hist["timestamp"] == linha.timestamp][0]
        esperado = hist["pH"].iloc[idx_real - (2 - 1)]
        assert linha.previsto == pytest.approx(esperado)
        assert linha.real == pytest.approx(hist["pH"].iloc[idx_real])


class _ScalerIdentidade:
    def transform(self, x):
        return x


def test_features_da_api_sao_carregaveis(db):
    from src.config import features_de
    from src.data_loader import load_plant_features, plantas_api

    gravar_lote(normalizar(_bruto(3)))
    plantas = plantas_api()
    assert not plantas.empty

    hist = load_plant_features(int(plantas["id_plant"].iloc[0]), features_de("api"))
    assert len(hist) == 3
    assert list(hist.columns) == ["timestamp", "pH", "EC", "temperature", "air_temp"]
    assert hist["temperature"].iloc[0] == pytest.approx(21.0)
    assert hist["air_temp"].iloc[0] == pytest.approx(26.0)
