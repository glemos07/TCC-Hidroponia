"""
Cliente HTTP da API de sensores (servidor/servidor_api.py).

A API e alimentada pelo ESP32 e expoe a telemetria bruta em `/v1/...`.
Este modulo isola todo o acesso HTTP do restante do sistema: nenhuma
outra parte do app conhece URLs ou nomes de campo da API.

Traducao de nomes aplicada em `renomear_para_features`:

    API (ESP32)       ->  modelo / banco
    ----------------      ---------------------------
    ph                ->  pH
    ec                ->  EC
    temp_agua         ->  temperature   (solucao nutritiva)
    temp_ambiente     ->  air_temp      (ar da estufa)

Uso:
    from src.api_client import ServidorSensores
    api = ServidorSensores()
    api.status()
    df = api.baixar_tudo(id_fazenda="fazenda-01")
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Iterable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import (
    API_BASE_URL,
    API_PAGE_SIZE,
    API_SENSOR_MAP,
    API_TIMEOUT,
)

#: Fuso usado pela API (servidor_api.py grava ISO-8601 com offset -03:00).
TZ_BR = "America/Sao_Paulo"

#: Colunas numericas devolvidas pela API.
COLUNAS_SENSOR = list(API_SENSOR_MAP)              # ph, ec, temp_agua, temp_ambiente
RENOMEIO = {campo: feature for campo, (_, _, feature) in API_SENSOR_MAP.items()}

COLUNAS_PADRAO = ["id", "id_fazenda", "id_planta", "nome_planta", "timestamp",
                  *COLUNAS_SENSOR, "sensores_simulados", "recebido_em"]


class ApiIndisponivel(RuntimeError):
    """A API nao respondeu, respondeu erro HTTP ou devolveu corpo invalido."""


class ServidorSensores:
    """Cliente somente-leitura (mais o POST de teste) do servidor FastAPI."""

    def __init__(self, base_url: str | None = None, timeout: float | None = None,
                 tentativas: int = 3):
        self.base_url = (base_url or API_BASE_URL).rstrip("/")
        self.timeout = timeout if timeout is not None else API_TIMEOUT

        self._sessao = requests.Session()
        # repete apenas falhas transitorias; erros 4xx sobem na hora
        politica = Retry(
            total=tentativas,
            backoff_factor=0.4,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
        )
        adaptador = HTTPAdapter(max_retries=politica)
        self._sessao.mount("http://", adaptador)
        self._sessao.mount("https://", adaptador)

    # ------------------------------------------------------------------ HTTP

    def _get(self, rota: str, params: dict | None = None) -> Any:
        url = f"{self.base_url}{rota}"
        try:
            r = self._sessao.get(url, params=_limpar(params), timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            raise ApiIndisponivel(f"GET {url} falhou: {e}") from e
        except ValueError as e:
            raise ApiIndisponivel(f"GET {url} devolveu JSON invalido: {e}") from e

    # -------------------------------------------------------------- consulta

    def status(self) -> dict:
        """
        Health check: confirma conectividade e total de leituras no servidor.

        A chave do contador mudou entre versoes do servidor (`leituras` nas
        builds publicadas, `total_leituras` no codigo atual), entao o retorno e
        normalizado aqui e sempre traz `total_leituras`.
        """
        corpo = self._get("/v1/status")
        if "total_leituras" not in corpo:
            corpo["total_leituras"] = corpo.get("leituras", 0)
        return corpo

    def online(self) -> bool:
        try:
            return self.status().get("STATUS") == "ok"
        except ApiIndisponivel:
            return False

    def plantas(self, id_fazenda: str) -> pd.DataFrame:
        """Plantas ativas da fazenda, com total de amostras e ultima leitura."""
        dados = self._get(f"/v1/fazendas/{id_fazenda}/plantas").get("plantas", [])
        df = pd.DataFrame(dados)
        if not df.empty:
            df["ultima_leitura"] = _para_datetime(df["ultima_leitura"])
        return df

    def ultima_leitura(self, id_planta: str | None = None) -> dict | None:
        """Leitura mais recente (para o painel em tempo real). None se nao houver."""
        try:
            return self._get("/v1/leituras/ultima", {"id_planta": id_planta})
        except ApiIndisponivel as e:
            if "404" in str(e):
                return None
            raise

    def estatisticas(self, id_fazenda: str | None = None, id_planta: str | None = None,
                     desde: datetime | None = None, ate: datetime | None = None) -> dict:
        """Media/minima/maxima de cada sensor no escopo informado."""
        return self._get("/v1/estatisticas", {
            "id_fazenda": id_fazenda, "id_planta": id_planta,
            "desde": _iso(desde), "ate": _iso(ate),
        })

    def leituras(self, id_planta: str | None = None, desde: datetime | None = None,
                 limite: int = 100, offset: int = 0) -> pd.DataFrame:
        """
        Pagina unica de `/v1/leituras` (ordem decrescente de timestamp).
        Adequada para previa na tela; para carga completa use `baixar_tudo`.
        """
        corpo = self._get("/v1/leituras", {
            "id_planta": id_planta, "desde": _iso(desde),
            "limite": min(limite, API_PAGE_SIZE), "offset": offset,
        })
        return normalizar(pd.DataFrame(corpo.get("dados", [])))

    def total_leituras(self, id_planta: str | None = None,
                       desde: datetime | None = None) -> int:
        corpo = self._get("/v1/leituras", {
            "id_planta": id_planta, "desde": _iso(desde), "limite": 1, "offset": 0,
        })
        return int(corpo.get("total", 0))

    def baixar_tudo(self, id_fazenda: str | None = None, id_planta: str | None = None,
                    desde: datetime | None = None, ate: datetime | None = None) -> pd.DataFrame:
        """
        Historico completo via `/v1/exportar/leituras` (CSV, ordem cronologica
        crescente, sem paginacao).

        Preferido a `/v1/leituras` para ingestao: aquele endpoint ordena por
        timestamp DESC e pagina por offset, de modo que leituras chegando
        durante o download deslocariam as paginas seguintes.
        """
        url = f"{self.base_url}/v1/exportar/leituras"
        params = _limpar({
            "formato": "csv", "id_fazenda": id_fazenda, "id_planta": id_planta,
            "desde": _iso(desde), "ate": _iso(ate),
        })
        try:
            r = self._sessao.get(url, params=params, timeout=self.timeout * 6)
            r.raise_for_status()
            texto = r.content.decode("utf-8-sig")     # a API prefixa BOM p/ Excel
        except requests.RequestException as e:
            raise ApiIndisponivel(f"GET {url} falhou: {e}") from e

        if not texto.strip():
            return normalizar(pd.DataFrame())
        return normalizar(pd.read_csv(io.StringIO(texto)))

    def enviar_leitura(self, leitura: dict) -> dict:
        """
        POST `/v1/leituras`. Existe para testes de ponta a ponta sem o ESP32
        ligado; em producao quem publica e o microcontrolador.
        """
        url = f"{self.base_url}/v1/leituras"
        try:
            r = self._sessao.post(url, json=leitura, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            raise ApiIndisponivel(f"POST {url} falhou: {e}") from e


# ------------------------------------------------------------------ helpers

def normalizar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza um lote de leituras cru da API: colunas garantidas, timestamp
    datetime sem fuso (horario de Sao Paulo), sensores como float e ordem
    cronologica crescente.
    """
    if df.empty:
        return pd.DataFrame(columns=COLUNAS_PADRAO)

    df = df.copy()
    for col in COLUNAS_PADRAO:
        if col not in df.columns:
            df[col] = pd.NA

    df["timestamp"] = _para_datetime(df["timestamp"])
    for col in COLUNAS_SENSOR:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return (df[COLUNAS_PADRAO]
            .dropna(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True))


def renomear_para_features(df: pd.DataFrame) -> pd.DataFrame:
    """Converte os nomes de campo da API para os nomes usados pelos modelos."""
    return df.rename(columns=RENOMEIO)


def _para_datetime(serie: pd.Series) -> pd.Series:
    """ISO-8601 com offset -> datetime local (sem fuso), como o resto do app usa."""
    dt = pd.to_datetime(serie, errors="coerce", utc=True)
    return dt.dt.tz_convert(TZ_BR).dt.tz_localize(None)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _limpar(params: dict | None) -> dict:
    """Remove chaves nulas para nao enviar `?x=None` na query string."""
    return {k: v for k, v in (params or {}).items() if v is not None and v != ""}


def resumo_simulados(valores: Iterable[Any]) -> str:
    """
    Agrega a coluna `sensores_simulados` (o ESP32 marca ali quais sensores
    estavam em modo simulado) num texto curto para exibicao.
    """
    vistos: list[str] = []
    for v in valores:
        if not v or (isinstance(v, float) and pd.isna(v)):
            continue
        for item in str(v).split(","):
            item = item.strip()
            if item and item.lower() not in ("false", "none") and item not in vistos:
                vistos.append(item)
    return ", ".join(vistos) if vistos else "nenhum"
