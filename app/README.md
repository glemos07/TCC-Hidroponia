# TCC UNIP 2026 — Hidroponia Inteligente

Sistema baseado em IoT e IA para monitoramento e previsão de pH e condutividade
elétrica (CE) em cultivos hidropônicos, empregando modelos de séries temporais
(ARIMA e LSTM multivariado).

Autores: Elber Jose Silva Nascimento (F3565I6) e Guilherme Lemos Freitas (G853BH3).
Orientador: Prof. Marco Gomes.

## Estrutura

```
app/
├── db/
│   └── schema.sql              # DDL das 10 entidades
├── src/
│   ├── config.py               # caminhos, constantes, seeds
│   ├── database.py             # camada SQLite
│   ├── synthetic_dataset.py    # gerador de dados sintéticos realistas
│   ├── seed.py                 # ETL: schema + carga
│   ├── data_loader.py          # séries temporais a partir do banco
│   ├── metrics.py              # RMSE, MAE, MAPE, Diebold-Mariano
│   ├── model_arima.py          # baseline univariado
│   ├── model_lstm.py           # LSTM multivariado (Keras)
│   └── evaluate.py             # Quadro 6 + Gráfico 3
├── streamlit_app/
│   ├── Home.py                 # tela inicial
│   └── pages/                  # RF01–RF07 + previsão
├── tests/                      # CT01–CT06
├── data/                       # CSVs gerados
├── db/hidroponia.sqlite3       # banco criado pelo seed
├── models/artifacts/           # modelos treinados
└── reports/                    # métricas + gráficos
```

## 1. Pré-requisitos

- **Python 3.11 ou 3.12** (https://www.python.org/downloads/)
- Durante a instalação, marque **Add Python to PATH**.

Verifique no PowerShell:

```powershell
python --version
```

## 2. Ambiente virtual e dependências

Do diretório `app/`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> Se o PowerShell bloquear o Activate.ps1, execute uma vez:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## 3. Popular o banco (seed)

### Opção A — Dataset real do Kaggle (recomendado para o TCC)

1. Baixe o dataset em https://www.kaggle.com/datasets/itsmonir31/hydroponics-datasets/data
2. Extraia o(s) CSV para `data/kaggle/`
3. Rode:
   ```powershell
   python -m src.seed
   ```

O seed detecta o CSV, aplica limpeza (nulos, duplicatas, tipos) — Quadro 1
do documento — mapeia colunas por sinônimos e popula o banco.

### Opção B — Dataset sintético (para demonstração sem internet)

Basta rodar `python -m src.seed` sem colocar nada em `data/kaggle/`.
O gerador sintético reprodutível (seed=42) cria a série alinhada aos
volumes do Quadro 2.

## 4. Treinar e avaliar os modelos

```powershell
python -m src.evaluate
```

Gera:
- `reports/metrics.json` — RMSE, MAE, MAPE + estatística Diebold-Mariano
- `reports/quadro6.txt` — Quadro 6 do documento
- `reports/grafico3_pH.png` e `reports/grafico3_EC.png` — Gráfico 3

## 5. Rodar o protótipo Streamlit

```powershell
streamlit run streamlit_app/Home.py
```

Abre no navegador as 7 telas dos requisitos funcionais:

| Código | Tela                              |
|--------|-----------------------------------|
| RF01   | Cadastrar planta                  |
| RF02   | Cadastrar ambiente                |
| RF03   | Registrar monitoramento           |
| RF04   | Consultar histórico               |
| RF05   | Registrar produção                |
| RF06   | Painel de indicadores             |
| RF07   | Importar dados iniciais           |

Uma oitava tela demonstra a previsão de pH/CE com ARIMA vs LSTM.

## 6. Executar os casos de teste

```powershell
pytest
```

Cobre CT01–CT06 conforme seção 5.9 do documento e testes das métricas.

## Alinhamento com o documento

| Documento                               | Implementação                                |
|-----------------------------------------|----------------------------------------------|
| 10 entidades (APÊNDICE A)               | `db/schema.sql` + `src/database.py`          |
| Quadro 1 (tratamento do dataset)        | `src/synthetic_dataset.py` + `src/seed.py`   |
| Quadro 2 (volumes)                      | Alvos: 120 plants, 1420 monitoring, 5630 sensor_data, 32 farm |
| Quadro 4 (estatísticas descritivas)     | Faixas simuladas: pH 5.42–6.78, CE 1.12–2.87, temp 18.4–27.9, umid 54.1–82.3 |
| Quadro 6 (ARIMA vs LSTM)                | `src/evaluate.py` → `reports/quadro6.txt`    |
| Gráfico 3 (observado vs previsto)       | `reports/grafico3_pH.png`                    |
| CT01–CT06 (Quadro 5)                    | `tests/test_casos_de_teste.py`               |
| Faixas ideais pH 5.5–6.5, CE 1.0–3.5    | `src/config.py` + alertas na RF03            |
| Hipótese H1 (LSTM < ARIMA em RMSE)      | Verificável em `reports/quadro6.txt`         |
| Teste Diebold-Mariano α=5%              | `src/metrics.py`                             |

## Observações

- **Dados**: por opção metodológica documentada (seção 4.4), o projeto usa um
  gerador sintético reprodutível (seed fixa). O loader pode ser adaptado a
  qualquer CSV com colunas `timestamp, pH, EC, temperature, humidity` para
  usar dados reais do Kaggle no futuro.
- **SGBD**: SQLite conforme seção 6.4 (portátil, sem servidor). O schema é
  compatível com PostgreSQL/MySQL para migração futura.
- **Modelos**: ARIMA usa refit expansivo em janela; LSTM tem 2 camadas
  recorrentes (64→32) + dropout, janela de 24 passos.
