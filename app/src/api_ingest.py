"""
ETL: API de sensores (ESP32) -> modelo relacional do TCC.

Converte a tabela plana `leituras` do servidor FastAPI para as entidades do
Apendice A (PLANTS, AMBIENT, SENSOR, MONITORING, SENSOR_DATA), de modo que
as telas de historico/indicadores e os modelos preditivos enxerguem o dado
real exatamente como enxergam o dado do seed.

    leituras (API)                    modelo relacional
    ------------------------------    -------------------------------------
    id_planta + nome_planta       ->  plants            (1 linha por planta)
    id_fazenda                    ->  ambient           (1 linha por fazenda)
    timestamp                     ->  monitoring.dt_register
    ph / ec / temp_agua /
      temp_ambiente               ->  sensor_data (1 linha por sensor lido)

A ingestao e idempotente: a tabela de controle `api_leitura` guarda o id
remoto de cada leitura ja importada, entao sincronizar duas vezes nao
duplica monitoramentos.

Uso:
    python -m src.api_ingest                       # sincroniza tudo
    python -m src.api_ingest --fazenda fazenda-01  # so uma fazenda
    python -m src.api_ingest --completo            # ignora o marcador e rele tudo
    python -m src.api_ingest --exportar            # salva o historico num CSV versionavel
    python -m src.api_ingest --arquivo             # reconstroi o banco do CSV, sem rede
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from src.api_client import TZ_BR as TZ_SNAPSHOT, ServidorSensores
from src.config import (
    API_FAIXAS_VALIDAS,
    API_ID_FAZENDA,
    API_SENSOR_MAP,
    DATA_DIR,
    SCHEMA_API_SQL,
)
from src.database import connect

#: Status preferido para plantas criadas pela ingestao. Faz parte do catalogo
#: carregado por `src.seed`; so e criado se o catalogo estiver vazio.
STATUS_PADRAO = "Crescimento"


@dataclass
class RelatorioSync:
    """Resumo de uma sincronizacao, exibido na tela e no terminal."""
    recebidas: int = 0
    novas: int = 0
    ja_importadas: int = 0
    plantas_criadas: list[str] = field(default_factory=list)
    medidas_inseridas: int = 0
    medidas_fora_faixa: dict[str, int] = field(default_factory=dict)
    descartadas: int = 0
    primeiro_ts: datetime | None = None
    ultimo_ts: datetime | None = None

    def as_dict(self) -> dict:
        fora = ", ".join(f"{campo}={n}" for campo, n in sorted(self.medidas_fora_faixa.items()))
        return {
            "leituras recebidas":   self.recebidas,
            "leituras novas":       self.novas,
            "ja importadas":        self.ja_importadas,
            "plantas criadas":      ", ".join(self.plantas_criadas) or "nenhuma",
            "medidas em sensor_data": self.medidas_inseridas,
            "medidas fora da faixa": fora or "nenhuma",
            "leituras descartadas": self.descartadas,
            "primeira leitura":     _fmt(self.primeiro_ts),
            "ultima leitura":       _fmt(self.ultimo_ts),
        }


# --------------------------------------------------------------------- schema

def aplicar_schema_api() -> None:
    """Cria as tabelas de controle da ingestao, se ainda nao existirem."""
    with connect() as conn:
        conn.executescript(SCHEMA_API_SQL.read_text(encoding="utf-8"))


def _descartar_vinculos_orfaos(conn: sqlite3.Connection) -> None:
    """
    Reexecutar `src.seed` recria (DROP/CREATE) as 10 entidades e zera os ids,
    deixando o rastreio da API apontando para linhas inexistentes. Detecta esse
    caso e limpa o rastreio, para que a proxima sincronizacao reimporte tudo.
    """
    pendentes = conn.execute("SELECT COUNT(*) FROM api_leitura").fetchone()[0]
    if not pendentes:
        return
    validas = conn.execute(
        "SELECT COUNT(*) FROM api_leitura a "
        "JOIN monitoring m ON m.id = a.id_monitoring"
    ).fetchone()[0]
    if validas == 0:
        conn.execute("DELETE FROM api_leitura")
        conn.execute("DELETE FROM api_planta")


# ------------------------------------------------------------------ dimensoes

def _garantir_sensores(conn: sqlite3.Connection) -> dict[str, int]:
    """Garante uma linha em SENSOR para cada grandeza publicada pela API."""
    ids: dict[str, int] = {}
    for campo, (tipo, unidade, _feature) in API_SENSOR_MAP.items():
        conn.execute(
            "INSERT OR IGNORE INTO sensor (sensor_type, unit, description) VALUES (?,?,?)",
            (tipo, unidade, f"Sensor {tipo} (ESP32, campo '{campo}' da API)"),
        )
        ids[campo] = conn.execute(
            "SELECT id FROM sensor WHERE sensor_type = ?", (tipo,)
        ).fetchone()[0]
    return ids


def _garantir_status(conn: sqlite3.Connection) -> int:
    """
    Status atribuido a uma planta descoberta pelos sensores. Reaproveita o
    catalogo carregado pelo seed em vez de inventar um valor novo; so cria
    STATUS_PADRAO se o catalogo estiver vazio.
    """
    linha = conn.execute(
        "SELECT id FROM farm_status WHERE status_name = ?", (STATUS_PADRAO,)
    ).fetchone()
    if linha:
        return linha[0]

    linha = conn.execute("SELECT id FROM farm_status ORDER BY id LIMIT 1").fetchone()
    if linha:
        return linha[0]

    cur = conn.execute("INSERT INTO farm_status (status_name) VALUES (?)", (STATUS_PADRAO,))
    return cur.lastrowid


def _garantir_ambiente(conn: sqlite3.Connection, id_fazenda: str) -> int:
    """Uma linha em AMBIENT por fazenda reportada pelo ESP32."""
    nome = f"Fazenda {id_fazenda}"
    linha = conn.execute("SELECT id FROM ambient WHERE nm_ambient = ?", (nome,)).fetchone()
    if linha:
        return linha[0]
    cur = conn.execute(
        "INSERT INTO ambient (nm_ambient, tp_ambient, descriptions) VALUES (?,?,?)",
        (nome, "Hidroponia IoT", f"Criado pela ingestao da API (id_fazenda={id_fazenda})"),
    )
    return cur.lastrowid


def _garantir_planta(conn: sqlite3.Connection, id_planta: str, nome_planta: str,
                     id_fazenda: str, relatorio: RelatorioSync,
                     dt_plantio: date | None = None) -> tuple[int, int]:
    """
    Resolve `id_planta` (texto, vindo do ESP32) para (plants.id, ambient.id),
    criando as linhas na primeira vez que a planta aparece.

    `dt_plantio` recebe a data da leitura mais antiga do lote: e a melhor
    estimativa disponivel de quando a planta entrou no sistema. Usar a data de
    hoje deixaria o cadastro mais novo que o proprio historico dela.
    """
    id_ambient = _garantir_ambiente(conn, id_fazenda)

    vinculo = conn.execute(
        "SELECT id_plant FROM api_planta WHERE id_planta = ?", (id_planta,)
    ).fetchone()
    if vinculo:
        existe = conn.execute("SELECT 1 FROM plants WHERE id = ?", (vinculo[0],)).fetchone()
        if existe:
            return vinculo[0], id_ambient
        conn.execute("DELETE FROM api_planta WHERE id_planta = ?", (id_planta,))

    rotulo = (nome_planta or "").strip() or id_planta
    plantio = (dt_plantio or datetime.now().date()).isoformat()
    cur = conn.execute(
        "INSERT INTO plants (name_plant, diversity, dt_planting, status_id) VALUES (?,?,?,?)",
        (rotulo, "Nao informado", plantio, _garantir_status(conn)),
    )
    id_plant = cur.lastrowid
    conn.execute(
        "INSERT OR REPLACE INTO api_planta "
        "(id_planta, id_fazenda, nome_planta, id_plant, id_ambient, criado_em) "
        "VALUES (?,?,?,?,?,?)",
        (id_planta, id_fazenda, rotulo, id_plant, id_ambient, datetime.now().isoformat()),
    )
    relatorio.plantas_criadas.append(f"{rotulo} ({id_planta})")
    return id_plant, id_ambient


# ----------------------------------------------------------------- marcadores

def ultima_leitura_importada(id_planta: str | None = None) -> datetime | None:
    """Timestamp da leitura mais recente ja ingerida (base da carga incremental)."""
    aplicar_schema_api()
    sql = "SELECT MAX(timestamp) FROM api_leitura"
    params: tuple = ()
    if id_planta:
        sql += " WHERE id_planta = ?"
        params = (id_planta,)
    with connect() as conn:
        valor = conn.execute(sql, params).fetchone()[0]
    return pd.to_datetime(valor).to_pydatetime() if valor else None


def resumo_ingestao() -> pd.DataFrame:
    """Plantas ja ingeridas, com volume e janela temporal de cada uma."""
    aplicar_schema_api()
    sql = """
    SELECT p.id_planta        AS id_planta,
           p.nome_planta      AS nome_planta,
           p.id_fazenda       AS id_fazenda,
           p.id_plant         AS id_plant,
           COUNT(l.id_remoto) AS leituras,
           MIN(l.timestamp)   AS primeira,
           MAX(l.timestamp)   AS ultima
    FROM   api_planta p
    LEFT   JOIN api_leitura l ON l.id_planta = p.id_planta
    GROUP  BY p.id_planta
    ORDER  BY p.nome_planta
    """
    with connect() as conn:
        return pd.read_sql_query(sql, conn)


# ----------------------------------------------------------------- ingestao

def sincronizar(api: ServidorSensores | None = None, id_fazenda: str | None = None,
                id_planta: str | None = None, incremental: bool = True,
                margem_min: int = 5) -> RelatorioSync:
    """
    Baixa as leituras da API e grava as inexistentes no modelo relacional.

    incremental: baixa apenas a partir da ultima leitura ja importada
                 (menos `margem_min` minutos, para cobrir pacotes do ESP32
                 que chegaram atrasados e ficaram para tras).
    """
    api = api or ServidorSensores()
    fazenda = id_fazenda if id_fazenda is not None else (API_ID_FAZENDA or None)

    desde = None
    if incremental:
        ultimo = ultima_leitura_importada(id_planta)
        if ultimo:
            desde = ultimo - timedelta(minutes=margem_min)

    lote = api.baixar_tudo(id_fazenda=fazenda, id_planta=id_planta, desde=desde)
    return gravar_lote(lote)


#: Instantaneo versionado do historico bruto, para reconstruir o banco sem rede.
ARQUIVO_SNAPSHOT = DATA_DIR / "sensores" / "leituras.csv"


def exportar_snapshot(api: ServidorSensores | None = None,
                      destino: Path | None = None,
                      id_fazenda: str | None = None) -> tuple[Path, int]:
    """
    Baixa o historico completo e grava em CSV versionavel.

    O banco SQLite nao vai para o repositorio (binario, muda a cada
    sincronizacao e nao da merge entre duas pessoas). Este CSV cumpre o papel
    de copia de seguranca: e texto, versiona bem e permite reconstruir o banco
    inteiro sem depender do servidor estar no ar.
    """
    api = api or ServidorSensores()
    destino = destino or ARQUIVO_SNAPSHOT
    fazenda = id_fazenda if id_fazenda is not None else (API_ID_FAZENDA or None)

    lote = api.baixar_tudo(id_fazenda=fazenda)
    destino.parent.mkdir(parents=True, exist_ok=True)

    # grava com o fuso explicito, do mesmo jeito que a API devolve, para que
    # reler o arquivo passe pela mesma normalizacao e de o mesmo resultado
    saida = lote.copy()
    saida["timestamp"] = (saida["timestamp"].dt.tz_localize(TZ_SNAPSHOT)
                          .dt.strftime("%Y-%m-%dT%H:%M:%S%z"))
    saida.to_csv(destino, index=False, encoding="utf-8")
    return destino, len(saida)


def carregar_snapshot(origem: Path | None = None) -> pd.DataFrame:
    """Le o instantaneo em CSV e devolve o lote normalizado, pronto para gravar."""
    from src.api_client import normalizar

    origem = origem or ARQUIVO_SNAPSHOT
    if not origem.exists():
        raise FileNotFoundError(
            f"Instantaneo nao encontrado em {origem}. Gere com:\n"
            "    python -m src.api_ingest --exportar"
        )
    return normalizar(pd.read_csv(origem))


def gravar_lote(lote: pd.DataFrame) -> RelatorioSync:
    """
    Persiste um lote ja normalizado (saida de `api_client.normalizar`).
    Separado de `sincronizar` para permitir teste sem servidor no ar.
    """
    aplicar_schema_api()
    rel = RelatorioSync(recebidas=len(lote))
    if lote.empty:
        return rel

    rel.primeiro_ts = lote["timestamp"].min().to_pydatetime()
    rel.ultimo_ts = lote["timestamp"].max().to_pydatetime()

    with connect() as conn:
        _descartar_vinculos_orfaos(conn)
        sensores = _garantir_sensores(conn)

        for (id_planta, id_fazenda), grupo in lote.groupby(["id_planta", "id_fazenda"], sort=False):
            id_plant, id_ambient = _garantir_planta(
                conn, str(id_planta), _primeiro_nome(grupo), str(id_fazenda), rel,
                dt_plantio=grupo["timestamp"].min().date(),
            )
            rel.novas += _gravar_planta(conn, grupo, id_plant, id_ambient, sensores, rel)

    rel.ja_importadas = rel.recebidas - rel.novas
    return rel


def _gravar_planta(conn: sqlite3.Connection, grupo: pd.DataFrame, id_plant: int,
                   id_ambient: int, sensores: dict[str, int],
                   rel: RelatorioSync) -> int:
    """Insere os monitoramentos de uma planta, pulando os ja importados."""
    id_planta = str(grupo["id_planta"].iloc[0])

    # Dois criterios de "ja importado". O id remoto e o natural, mas builds
    # antigas do servidor aceitaram a mesma leitura duas vezes com ids
    # diferentes (sem a UNIQUE de hoje), e nesse caso o par (planta, instante)
    # e quem identifica a leitura. Sem o segundo criterio, cada carga repetida
    # criaria um monitoramento duplicado.
    importados_id = {
        linha[0] for linha in conn.execute(
            "SELECT id_remoto FROM api_leitura WHERE id_planta = ?", (id_planta,))
    }
    importados_ts = {
        linha[0] for linha in conn.execute(
            "SELECT timestamp FROM api_leitura WHERE id_planta = ?", (id_planta,))
    }

    novas = 0
    medidas: list[tuple] = []
    rastreio: list[tuple] = []
    agora = datetime.now().isoformat()

    for linha in grupo.itertuples(index=False):
        id_remoto = _para_int(linha.id)
        if id_remoto is not None and id_remoto in importados_id:
            continue
        ts = pd.Timestamp(linha.timestamp).to_pydatetime()
        ts_iso = ts.isoformat(sep=" ")
        if ts_iso in importados_ts:          # cobre repeticoes dentro do lote
            continue

        # a leitura so vira monitoramento se tiver ao menos um sensor valido
        valores = [(sensores[campo], valor)
                   for campo in API_SENSOR_MAP
                   if (valor := _validar(campo, getattr(linha, campo), rel)) is not None]
        if not valores:
            rel.descartadas += 1
            continue

        cur = conn.execute(
            "INSERT INTO monitoring (dt_register, id_plant, id_ambient, id_user, obs) "
            "VALUES (?,?,?,?,?)",
            (ts_iso, id_plant, id_ambient, None, _observacao(linha.sensores_simulados)),
        )
        id_monitoring = cur.lastrowid
        medidas.extend((id_monitoring, id_sensor, valor) for id_sensor, valor in valores)
        rastreio.append((id_remoto, id_planta, ts_iso, id_monitoring, agora))
        importados_ts.add(ts_iso)
        novas += 1

    if medidas:
        conn.executemany(
            "INSERT INTO sensor_data (id_monitoring, id_sensor, value) VALUES (?,?,?)",
            medidas,
        )
        rel.medidas_inseridas += len(medidas)
    if rastreio:
        # OR IGNORE, nunca OR REPLACE: substituir o rastreio de uma leitura
        # deixaria o monitoramento dela orfao e ele seria recriado na proxima
        # sincronizacao.
        conn.executemany(
            "INSERT OR IGNORE INTO api_leitura "
            "(id_remoto, id_planta, timestamp, id_monitoring, importado_em) VALUES (?,?,?,?,?)",
            rastreio,
        )
    return novas


# ------------------------------------------------------------------- helpers

def _validar(campo: str, valor, rel: RelatorioSync) -> float | None:
    """
    Converte a medida para float, descartando nulos e valores fisicamente
    impossiveis (ex.: pH 13.1 ou CE 25 mS/cm sao falha de sensor, nao cultivo).
    Retorna None quando a medida nao deve virar linha em SENSOR_DATA.
    """
    if pd.isna(valor):
        return None
    valor = float(valor)
    minimo, maximo = API_FAIXAS_VALIDAS[campo]
    if not (minimo <= valor <= maximo):
        rel.medidas_fora_faixa[campo] = rel.medidas_fora_faixa.get(campo, 0) + 1
        return None
    return valor


def _primeiro_nome(grupo: pd.DataFrame) -> str:
    nomes = grupo["nome_planta"].dropna().astype(str)
    nomes = nomes[nomes.str.strip() != ""]
    return nomes.iloc[0] if not nomes.empty else ""


def _observacao(simulados) -> str | None:
    if not simulados or (isinstance(simulados, float) and pd.isna(simulados)):
        return None
    texto = str(simulados).strip()
    if not texto or texto.lower() in ("false", "none", "nan"):
        return None
    return f"Sensores simulados: {texto}"[:255]


def _para_int(valor) -> int | None:
    try:
        return int(valor)
    except (TypeError, ValueError):
        return None


def _fmt(dt: datetime | None) -> str:
    return dt.strftime("%d/%m/%Y %H:%M:%S") if dt else "-"


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Sincroniza as leituras da API de sensores.")
    p.add_argument("--url", default=None, help="URL base da API (default: HIDRO_API_URL)")
    p.add_argument("--fazenda", default=None, help="Filtrar por id_fazenda")
    p.add_argument("--planta", default=None, help="Filtrar por id_planta")
    p.add_argument("--completo", action="store_true",
                   help="Ignora o marcador incremental e rele todo o historico")
    p.add_argument("--exportar", nargs="?", const="", metavar="CSV",
                   help="Baixa o historico completo para um CSV versionavel e sai")
    p.add_argument("--arquivo", nargs="?", const="", metavar="CSV",
                   help="Importa de um CSV local em vez da API (funciona sem rede)")
    args = p.parse_args()

    if args.exportar is not None:
        api = ServidorSensores(base_url=args.url)
        destino, n = exportar_snapshot(
            api, Path(args.exportar) if args.exportar else None, args.fazenda)
        print(f"{n} leituras gravadas em {destino}")
        return

    if args.arquivo is not None:
        origem = Path(args.arquivo) if args.arquivo else None
        lote = carregar_snapshot(origem)
        print(f"Lendo {len(lote)} leituras de {origem or ARQUIVO_SNAPSHOT}")
        rel = gravar_lote(lote)
        print("\nImportacao concluida")
        print("-" * 60)
        for chave, valor in rel.as_dict().items():
            print(f"  {chave:24s} {valor}")
        return

    api = ServidorSensores(base_url=args.url)
    print(f"API: {api.base_url}")
    print(f"Status: {api.status()}")

    rel = sincronizar(api, id_fazenda=args.fazenda, id_planta=args.planta,
                      incremental=not args.completo)

    print("\nSincronizacao concluida")
    print("-" * 60)
    for chave, valor in rel.as_dict().items():
        print(f"  {chave:24s} {valor}")


if __name__ == "__main__":
    main()
