-- Schema complementar: rastreio da ingestao vinda da API de sensores
-- TCC UNIP 2026 - Elber e Guilherme
--
-- Estas tabelas NAO fazem parte do modelo relacional do Quadro 2 (Apendice A):
-- sao tabelas de controle do ETL, criadas sob demanda pela primeira
-- sincronizacao com o servidor FastAPI e preservadas entre execucoes.
--
-- Nao declaram FOREIGN KEY de proposito: `schema.sql` recria (DROP/CREATE) as
-- 10 entidades a cada seed, e uma FK impediria esse recriar. A consistencia e
-- revalidada pelo proprio ETL (src/api_ingest.py), que recria o vinculo quando
-- a planta referenciada deixa de existir.

-- Correspondencia entre a planta identificada pelo ESP32 (texto) e a
-- linha correspondente em PLANTS/AMBIENT do modelo relacional.
CREATE TABLE IF NOT EXISTS api_planta (
    id_planta    TEXT    PRIMARY KEY,   -- id_planta devolvido pela API
    id_fazenda   TEXT    NOT NULL,
    nome_planta  TEXT,
    id_plant     INTEGER NOT NULL,      -- -> plants.id
    id_ambient   INTEGER NOT NULL,      -- -> ambient.id
    criado_em    TEXT    NOT NULL
);

-- Uma linha por leitura ja importada. Garante idempotencia: reexecutar a
-- sincronizacao nao duplica monitoramentos.
CREATE TABLE IF NOT EXISTS api_leitura (
    id_remoto     INTEGER PRIMARY KEY,  -- leituras.id na API
    id_planta     TEXT    NOT NULL,
    timestamp     TEXT    NOT NULL,
    id_monitoring INTEGER NOT NULL,     -- -> monitoring.id
    importado_em  TEXT    NOT NULL,
    UNIQUE (id_planta, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_api_leitura_planta ON api_leitura (id_planta, timestamp);
