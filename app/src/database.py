"""Camada de acesso ao SQLite (SGBD conforme seção 6.4 do documento)."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from src import config as _config
from src.config import SCHEMA_SQL


@contextmanager
def connect(path: Path | None = None):
    # resolve DB_PATH em tempo de chamada (permite monkeypatch em testes)
    if path is None:
        path = _config.DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def apply_schema() -> None:
    with connect() as conn:
        ddl = SCHEMA_SQL.read_text(encoding="utf-8")
        conn.executescript(ddl)


def insert_dataframe(table: str, df: pd.DataFrame, conn: sqlite3.Connection) -> None:
    if df.empty:
        return
    cols = list(df.columns)
    placeholders = ",".join(["?"] * len(cols))
    columns_sql = ",".join(cols)
    sql = f"INSERT INTO {table} ({columns_sql}) VALUES ({placeholders})"
    rows = [tuple(None if pd.isna(v) else v for v in row) for row in df.itertuples(index=False, name=None)]
    conn.executemany(sql, rows)


def fetch_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def execute(sql: str, params: tuple = ()) -> int:
    with connect() as conn:
        cur = conn.execute(sql, params)
        return cur.lastrowid
