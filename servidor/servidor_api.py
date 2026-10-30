import logging
import os
import sqlite3
import json
import csv
import io
from contextlib import asynccontextmanager, closing
from datetime import datetime
from typing import List, Optional, Union
from zoneinfo import ZoneInfo
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, field_validator

TZ_BR = ZoneInfo("America/Sao_Paulo")
ARQUIVO_DB = os.getenv("HIDRO_DB", "hidroponia_v3.db")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("hidroponia")

SCHEMA = """
CREATE TABLE IF NOT EXISTS leituras (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    id_fazenda            TEXT NOT NULL,
    id_planta             TEXT NOT NULL,
    nome_planta           TEXT,
    timestamp             TEXT NOT NULL,
    temp_agua             REAL,
    temp_ambiente         REAL,
    ph                    REAL,
    ec                    REAL,
    sensores_simulados    TEXT,
    recebido_em           TEXT NOT NULL,
    UNIQUE (id_fazenda, id_planta, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_planta_ts ON leituras (id_planta, timestamp);
"""

def conectar() -> sqlite3.Connection:
    """
    Cria e retorna uma conexão segura com o banco de dados SQLite.
    Aplica row_factory para retorno em formato de dicionário e define um timeout longo
    para concorrência em ambiente de produção (modo WAL).
    """
    db_dir = os.path.dirname(ARQUIVO_DB)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        
    conn = sqlite3.connect(ARQUIVO_DB, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn

@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação.
    Executa a criação das tabelas e otimiza o SQLite para modo WAL 
    (Write-Ahead Logging), melhorando a performance de leitura/escrita simultânea.
    """
    with closing(conectar()) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(SCHEMA)
    log.info("Banco pronto para produção: %s", os.path.abspath(ARQUIVO_DB))
    yield

app = FastAPI(
    title="API Hidroponia ESP32", 
    description="API RESTful para recebimento, persistência e exportação de dados telemétricos de sensores hidropônicos.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Leitura(BaseModel):
    """
    Modelo de validação de dados de entrada do ESP32.
    Converte flexivelmente falhas de leitura ("Falha") para valores nulos aceitáveis pelo banco.
    """
    model_config = ConfigDict(coerce_numbers_to_str=True)

    id_fazenda: str
    id_planta: str
    nome_planta: str = ""
    timestamp: datetime
    temp_agua: Optional[float] = None
    temp_ambiente: Optional[float] = None
    ph: Optional[float] = None
    ec: Optional[float] = None
    sensores_simulados: Union[List[str], bool, None] = None

    @field_validator("temp_agua", "temp_ambiente", "ph", "ec", mode="before")
    @classmethod
    def texto_vira_nulo(cls, v):
        if isinstance(v, str):
            try:
                return float(v.replace(",", "."))
            except ValueError:
                return None
        return v

def normalizar_timestamp(dt: datetime) -> datetime:
    """Garante que objetos datetime usem o fuso horário padrão do sistema (America/Sao_Paulo)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=TZ_BR)
    return dt.astimezone(TZ_BR)

@app.exception_handler(RequestValidationError)
async def erro_validacao(request: Request, exc: RequestValidationError):
    """Captura erros de formatação JSON e devolve respostas padronizadas sem derrubar a API."""
    corpo = (await request.body()).decode(errors="replace")[:500]
    origem = request.client.host if request.client else "?"
    log.warning("Erro HTTP 422 de %s | erros=%s | corpo=%s", origem, exc.errors(), corpo)
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})

@app.get("/v1/status", summary="Verificar saúde da API")
def verificar_status():
    """
    Endpoint de checagem de saúde (Health Check).
    Consultado pelo ESP32 antes de submeter os dados para confirmar conectividade.
    """
    try:
        with closing(conectar()) as conn:
            total = conn.execute("SELECT COUNT(*) FROM leituras").fetchone()[0]
        return {"STATUS": "ok", "hora_servidor": datetime.now(TZ_BR).strftime("%d/%m/%Y %H:%M:%S"), "total_leituras": total}
    except sqlite3.Error as e:
        log.error("Erro no Health Check: %s", e)
        raise HTTPException(status_code=500, detail="Banco de dados inacessível.")

@app.post("/v1/leituras", status_code=status.HTTP_201_CREATED, summary="Registrar nova leitura")
def receber_leitura(leitura: Leitura, request: Request):
    """
    Recebe um pacote de telemetria do ESP32 e o insere no banco de dados SQLite.
    Ignora duplicatas exatas baseadas em (id_fazenda, id_planta, timestamp).
    """
    ts = normalizar_timestamp(leitura.timestamp)
    simulados = leitura.sensores_simulados
    simulados_str = ",".join(simulados) if isinstance(simulados, list) else (str(simulados) if simulados is not None else None)
    
    try:
        with closing(conectar()) as conn, conn:    
            conn.execute(
                """INSERT OR IGNORE INTO leituras
                   (id_fazenda, id_planta, nome_planta, timestamp, temp_agua, temp_ambiente, ph, ec,
                    sensores_simulados, recebido_em)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (leitura.id_fazenda, leitura.id_planta, leitura.nome_planta, ts.isoformat(),
                 leitura.temp_agua, leitura.temp_ambiente, leitura.ph, leitura.ec,
                 simulados_str, datetime.now(TZ_BR).isoformat()),
            )
            duplicada = conn.execute("SELECT changes()").fetchone()[0] == 0
    except sqlite3.Error:
        log.exception("Falha estrutural ao gravar leitura no banco de dados.")
        raise HTTPException(status_code=500, detail="Erro interno ao persistir dados.")

    log.info("%s %s | agua=%s amb=%s ph=%s ec=%s%s%s", request.client.host if request.client else "?",
             leitura.nome_planta, leitura.temp_agua, leitura.temp_ambiente, leitura.ph, leitura.ec,
             f" | SIMULADOS={simulados_str}" if simulados_str else "", " | DUPLICADA (ignorada)" if duplicada else "")
    
    return {"status": "UP", "duplicada": duplicada}

@app.get("/v1/leituras", summary="Listar leituras globais")
def listar_leituras(
    id_planta: Optional[str] = Query(None, description="Filtrar por identificador da planta"),
    desde: Optional[datetime] = Query(None, description="Filtrar dados a partir de uma data (ISO 8601)"),
    limite: int = Query(100, ge=1, le=1000, description="Quantidade máxima de registros retornados"),
    offset: int = Query(0, ge=0, description="Salto de registros para paginação"),
):
    """
    Retorna o histórico geral de leituras com paginação e ordenação cronológica decrescente.
    """
    filtros, params = [], []
    if id_planta:
        filtros.append("id_planta = ?")
        params.append(id_planta)
    if desde:
        filtros.append("timestamp >= ?")
        params.append(normalizar_timestamp(desde).isoformat())
        
    where = (" WHERE " + " AND ".join(filtros)) if filtros else ""
    
    try:
        with closing(conectar()) as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM leituras{where}", params).fetchone()[0]
            linhas = conn.execute(f"SELECT * FROM leituras{where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                                  params + [limite, offset]).fetchall()
        return {"total": total, "limite": limite, "offset": offset, "dados": [dict(r) for r in linhas]}
    except sqlite3.Error as e:
        log.error("Erro na consulta de leituras: %s", e)
        raise HTTPException(status_code=500, detail="Falha ao processar consulta no banco de dados.")

@app.get("/v1/leituras/ultima", summary="Obter leitura mais recente")
def ultima_leitura(id_planta: Optional[str] = Query(None, description="Identificador da planta (opcional)")):
    """
    Retorna o último registro capturado, útil para exibir o status em tempo real em um painel.
    """
    sql, params = "SELECT * FROM leituras", []
    if id_planta:
        sql += " WHERE id_planta = ?"
        params.append(id_planta)
        
    with closing(conectar()) as conn:
        r = conn.execute(sql + " ORDER BY timestamp DESC LIMIT 1", params).fetchone()
        
    if r is None:
        raise HTTPException(status_code=404, detail="Nenhuma leitura encontrada no histórico.")
    return dict(r)

@app.get("/v1/leituras/{leitura_id}", summary="Consultar leitura por ID")
def consultar_leitura(leitura_id: int):
    """
    Busca um registro específico na tabela utilizando o ID primário interno.
    """
    with closing(conectar()) as conn:
        linha = conn.execute("SELECT * FROM leituras WHERE id = ?", (leitura_id,)).fetchone()

    if linha is None:
        raise HTTPException(status_code=404, detail="Leitura não encontrada para o ID fornecido.")
    return dict(linha)

@app.get("/v1/fazendas/{id_fazenda}/leituras", summary="Listar leituras por fazenda")
def leituras_por_fazenda(
    id_fazenda: str,
    id_planta: str | None = Query(None, description="Filtrar por planta específica da fazenda"),
    desde: datetime | None = None,
    ate: datetime | None = None,
    limite: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """
    Retorna todo o histórico de leituras vinculado a uma fazenda específica,
    suportando filtros detalhados de planta e janela temporal.
    """
    filtros, params = ["id_fazenda = ?"], [id_fazenda]

    if id_planta:
        filtros.append("id_planta = ?")
        params.append(id_planta)
    if desde:
        filtros.append("timestamp >= ?")
        params.append(normalizar_timestamp(desde).isoformat())
    if ate:
        filtros.append("timestamp <= ?")
        params.append(normalizar_timestamp(ate).isoformat())

    where = " AND ".join(filtros)

    with closing(conectar()) as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM leituras WHERE {where}", params).fetchone()[0]
        linhas = conn.execute(
            f"SELECT * FROM leituras WHERE {where} ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?",
            params + [limite, offset],
        ).fetchall()

    return {
        "fazenda": id_fazenda,
        "total": total,
        "limite": limite,
        "offset": offset,
        "dados": [dict(linha) for linha in linhas],
    }

@app.get("/v1/fazendas/{id_fazenda}/plantas", summary="Listar plantas cadastradas na fazenda")
def listar_plantas(id_fazenda: str):
    """
    Agrupa os dados do histórico para exibir uma listagem de todas as plantas ativas
    em uma fazenda, informando o total de amostras lidas e a data da última leitura.
    """
    with closing(conectar()) as conn:
        linhas = conn.execute(
            """
            SELECT
                id_planta,
                MAX(nome_planta) AS nome_planta,
                COUNT(*) AS total_leituras,
                MAX(timestamp) AS ultima_leitura
            FROM leituras
            WHERE id_fazenda = ?
            GROUP BY id_planta
            ORDER BY nome_planta
            """,
            (id_fazenda,),
        ).fetchall()

    return {
        "fazenda": id_fazenda,
        "total_plantas": len(linhas),
        "plantas": [dict(linha) for linha in linhas],
    }

@app.get("/v1/estatisticas", summary="Gerar estatísticas de sensores")
def estatisticas(
    id_fazenda: str | None = None,
    id_planta: str | None = None,
    desde: datetime | None = None,
    ate: datetime | None = None,
):
    """
    Calcula de forma agregada as métricas (Média, Mínima, Máxima) de cada um dos sensores 
    (Água, Ar, pH, EC) baseado no escopo fornecido.
    """
    filtros, params = [], []

    if id_fazenda:
        filtros.append("id_fazenda = ?"); params.append(id_fazenda)
    if id_planta:
        filtros.append("id_planta = ?"); params.append(id_planta)
    if desde:
        filtros.append("timestamp >= ?"); params.append(normalizar_timestamp(desde).isoformat())
    if ate:
        filtros.append("timestamp <= ?"); params.append(normalizar_timestamp(ate).isoformat())

    where = " WHERE " + " AND ".join(filtros) if filtros else ""

    with closing(conectar()) as conn:
        resultado = conn.execute(
            f"""
            SELECT
                COUNT(*) AS total_leituras,
                AVG(temp_agua) AS media_temp_agua, MIN(temp_agua) AS min_temp_agua, MAX(temp_agua) AS max_temp_agua,
                AVG(temp_ambiente) AS media_temp_ambiente, MIN(temp_ambiente) AS min_temp_ambiente, MAX(temp_ambiente) AS max_temp_ambiente,
                AVG(ph) AS media_ph, MIN(ph) AS min_ph, MAX(ph) AS max_ph,
                AVG(ec) AS media_ec, MIN(ec) AS min_ec, MAX(ec) AS max_ec
            FROM leituras
            {where}
            """,
            params,
        ).fetchone()

    return dict(resultado)

@app.get("/v1/exportar/leituras", summary="Exportar dados para download")
def exportar_leituras(
    formato: str = Query("csv", pattern="^(csv|json)$", description="Formato desejado do download"),
    id_fazenda: str | None = None,
    id_planta: str | None = None,
    desde: datetime | None = None,
    ate: datetime | None = None,
):
    """
    Gera dinamicamente um arquivo estruturado (CSV ou JSON) para download, 
    permitindo que cientistas de dados ou produtores importem o histórico para análise.
    """
    filtros, params = [], []

    if id_fazenda:
        filtros.append("id_fazenda = ?"); params.append(id_fazenda)
    if id_planta:
        filtros.append("id_planta = ?"); params.append(id_planta)
    if desde:
        filtros.append("timestamp >= ?"); params.append(normalizar_timestamp(desde).isoformat())
    if ate:
        filtros.append("timestamp <= ?"); params.append(normalizar_timestamp(ate).isoformat())

    where = " WHERE " + " AND ".join(filtros) if filtros else ""

    with closing(conectar()) as conn:
        linhas = conn.execute(f"SELECT * FROM leituras {where} ORDER BY timestamp ASC, id ASC", params).fetchall()

    dados = [dict(linha) for linha in linhas]

    if formato == "json":
        conteudo = json.dumps(dados, ensure_ascii=False, indent=2)
        return StreamingResponse(
            iter([conteudo]),
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="leituras.json"'}
        )

    buffer = io.StringIO()
    buffer.write("\ufeff") # BOM para suporte Excel UTF-8
    campos = ["id", "id_fazenda", "id_planta", "nome_planta", "timestamp", 
              "temp_agua", "temp_ambiente", "ph", "ec", "sensores_simulados", "recebido_em"]

    writer = csv.DictWriter(buffer, fieldnames=campos)
    writer.writeheader()
    writer.writerows(dados)

    conteudo = buffer.getvalue()
    buffer.close()

    return StreamingResponse(
        iter([conteudo]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="leituras.csv"'}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)