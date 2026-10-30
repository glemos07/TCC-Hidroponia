# TCC UNIP 2026 — Hidroponia Inteligente

Sistema baseado em IoT e IA para monitoramento e previsão de pH e condutividade
elétrica (CE) em cultivos hidropônicos, empregando modelos de séries temporais
(ARIMA e LSTM multivariado).

Autores: Elber Jose Silva Nascimento (F3565I6) e Guilherme Lemos Freitas (G853BH3).
Orientador: Prof. Marco Gomes.

## Estrutura

```
servidor/
└── servidor_api.py             # API FastAPI que recebe a telemetria do ESP32

app/
├── db/
│   ├── schema.sql              # DDL das 10 entidades
│   └── schema_api.sql          # tabelas de controle da ingestão via API
├── src/
│   ├── config.py               # caminhos, constantes, catálogo de sensores
│   ├── database.py             # camada SQLite
│   ├── seed.py                 # schema + catálogos de referência
│   ├── data_loader.py          # séries temporais (sensores ou Kaggle)
│   ├── api_client.py           # cliente HTTP da API de sensores (ESP32)
│   ├── api_ingest.py           # ETL: leituras da API → modelo relacional
│   ├── metrics.py              # RMSE, MAE, MAPE, Diebold-Mariano
│   ├── model_arima.py          # baseline univariado
│   ├── model_lstm.py           # LSTM multivariado (Keras)
│   └── evaluate.py             # Quadro 6 + Gráfico 3
├── streamlit_app/
│   ├── Home.py                 # tela inicial
│   └── pages/                  # RF01–RF07 + previsão
├── tests/                      # CT01–CT06
├── data/kaggle/                # dataset público (comparação; fora do banco)
├── db/hidroponia.sqlite3       # banco: catálogos + leituras dos sensores
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

## 3. Inicializar o banco

```powershell
python -m src.seed
```

Cria as 10 entidades do Apêndice A e carrega **somente os catálogos de
referência**: os 5 status de cultivo, os 12 nutrientes e os 4 tipos de sensor
que o hardware realmente mede. As demais tabelas nascem vazias.

| Entidade | Origem dos registros |
|----------|----------------------|
| `farm_status`, `nutrient`, `sensor` | catálogo de referência (o seed carrega) |
| `monitoring`, `sensor_data` | sensores, via API (passo 4) |
| `plants`, `ambient` | sensores, ou cadastro manual em RF01/RF02 |
| `users`, `farm`, `monitoring_nutrient` | cadastro manual em RF05 e RF03 |

> **Não existe gerador de dados sintéticos no projeto.** Todo o histórico vem
> dos sensores. O dataset do Kaggle é opcional e serve só de base de comparação
> para os modelos (passo 5) — ele é lido direto do CSV e **não entra no banco**.

## 4. Consumir os dados reais dos sensores (API + ESP32)

O ESP32 publica a telemetria no servidor FastAPI (`servidor/servidor_api.py`),
que a guarda numa tabela plana `leituras`. A aplicação consome essa API e
converte as leituras para as entidades do Apêndice A, de onde os modelos
preditivos as leem como qualquer outro dado.

### 4.1 Subir o servidor

```powershell
python -m pip install -r ..\servidor\requirements.txt
python ..\servidor\servidor_api.py          # escuta em 0.0.0.0:8000
```

Documentação interativa dos endpoints em `http://localhost:8000/docs`.

### 4.2 Apontar a aplicação para o servidor

```powershell
$env:HIDRO_API_URL = "http://<ip-do-servidor>:8000"   # servidor publicado do projeto
$env:HIDRO_API_FAZENDA = "101"                        # opcional
```

> O endereço do servidor publicado não está versionado de propósito: a API é
> aberta (sem autenticação) e aceita `POST /v1/leituras`, então divulgá-la num
> repositório público permitiria que qualquer um lesse ou injetasse leituras.

Sem essas variáveis vale o padrão `http://localhost:8000` e todas as fazendas.
A URL também pode ser digitada na tela **10_Servidor_API**, que detecta sozinha
a fazenda ativa quando o filtro fica em branco.

> O `/v1/status` da instância publicada devolve a chave `leituras`, enquanto o
> `servidor_api.py` deste repositório devolve `total_leituras` — a instância no
> ar é de uma versão anterior. `api_client.status()` aceita as duas.

### 4.3 Importar as leituras

Pela interface: tela **10_Servidor_API** → botão *Sincronizar agora*.
Pelo terminal:

```powershell
python -m src.api_ingest                        # incremental (só o que é novo)
python -m src.api_ingest --fazenda fazenda-01   # restringe a uma fazenda
python -m src.api_ingest --completo             # relê todo o histórico
```

A ingestão é **idempotente**: a tabela `api_leitura` guarda o id remoto de cada
leitura já importada, então sincronizar duas vezes não duplica monitoramentos.
Uma leitura é reconhecida pelo id remoto **ou** pelo par (planta, instante) —
builds antigas do servidor, sem a restrição `UNIQUE` de hoje, aceitaram a mesma
leitura duas vezes com ids diferentes.

### 4.4 Tratamento aplicado na entrada (Quadro 1)

As mesmas regras de qualidade usadas no dataset Kaggle valem para a telemetria:

- **Faixas físicas** (`config.API_FAIXAS_VALIDAS`): pH 4,0–8,5; CE 0,5–4,0 mS/cm;
  temperatura da solução 15–35 °C; temperatura do ar 0–50 °C. Medidas fora
  disso são falha de sensor, não variação do cultivo — a medida é descartada,
  mas as demais da mesma leitura são mantidas. O relatório da sincronização
  informa quantas caíram, por sensor.
- **Leitura vazia**: se nenhum dos quatro sensores sobreviveu, a leitura não
  vira monitoramento.
- **Lacunas de coleta**: queda de energia ou de rede deixa buracos de horas na
  série. No treino, janelas do LSTM que atravessam uma interrupção maior que
  `MAX_GAP_FACTOR` × o intervalo mediano são descartadas, e a tela 8 avisa
  quando a janela usada na previsão contém uma dessas paradas.

Mapeamento aplicado:

| Campo da API    | Entidade / sensor no banco       | Feature do modelo |
|-----------------|----------------------------------|-------------------|
| `id_planta`     | `plants` (1 linha por planta)    | —                 |
| `id_fazenda`    | `ambient` (1 linha por fazenda)  | —                 |
| `timestamp`     | `monitoring.dt_register`         | —                 |
| `ph`            | `sensor` = *pH*                  | `pH`              |
| `ec`            | `sensor` = *EC*                  | `EC`              |
| `temp_agua`     | `sensor` = *Temperatura*         | `temperature`     |
| `temp_ambiente` | `sensor` = *Temperatura Ambiente*| `air_temp`        |

> A API **não tem sensor de umidade**, mas tem duas temperaturas. Por isso cada
> fonte tem seu próprio vetor de entrada (`config.FEATURE_SETS`) e seus próprios
> artefatos treinados — os modelos do Quadro 6 continuam válidos e intactos.

## 5. Treinar e avaliar os modelos

```powershell
python -m src.evaluate            # fonte "kaggle" (Quadro 6 do documento)
python -m src.evaluate api        # fonte "api"    (telemetria real do ESP32)
```

Gera:
- `reports/metrics.json` — RMSE, MAE, MAPE + estatística Diebold-Mariano
- `reports/quadro6.txt` — Quadro 6 do documento
- `reports/grafico3_pH.png` e `reports/grafico3_EC.png` — Gráfico 3

Com `api`, os arquivos recebem o sufixo `_api` (`quadro6_api.txt`,
`lstm_pH_api.keras`, …) e nada do Quadro 6 original é sobrescrito. A tela 8 usa
sempre os modelos `api`: previsão por planta lê o histórico do banco, que só
tem sensores. A comparação sobre o dataset público fica na tela 9.
O LSTM exige pelo menos `2 × (janela + horizonte)` leituras da planta —
com a janela padrão de 240 passos, são **600 leituras** no mínimo.

## 6. Rodar o protótipo Streamlit

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
| RF07   | Inicializar banco                 |

Telas adicionais:

- **8_Previsao_por_Planta** — previsão de pH/CE por planta, pelo LSTM treinado
  sobre as leituras dos sensores dela.
- **9_Comparativo_Modelos** — Quadro 6: ARIMA vs LSTM.
- **10_Servidor_API** — saúde do servidor FastAPI, telemetria recebida do ESP32
  e importação das leituras para o banco.

## 7. Executar os casos de teste

```powershell
pytest
```

Cobre CT01–CT06 conforme seção 5.9 do documento e testes das métricas.

## Alinhamento com o documento

| Documento                               | Implementação                                |
|-----------------------------------------|----------------------------------------------|
| 10 entidades (APÊNDICE A)               | `db/schema.sql` + `src/database.py`          |
| Quadro 1 (tratamento do dataset)        | `src/kaggle_loader.py` + `src/api_ingest.py` |
| Quadro 2 (volumes)                      | Medidos no banco após a ingestão dos sensores |
| Quadro 4 (estatísticas descritivas)     | Calculadas sobre as leituras reais na tela RF06 |
| Quadro 6 (ARIMA vs LSTM)                | `src/evaluate.py` → `reports/quadro6.txt`    |
| Gráfico 3 (observado vs previsto)       | `reports/grafico3_pH.png`                    |
| CT01–CT06 (Quadro 5)                    | `tests/test_casos_de_teste.py`               |
| Faixas ideais pH 5.5–6.5, CE 1.0–3.5    | `src/config.py` + alertas na RF03            |
| Hipótese H1 (LSTM < ARIMA em RMSE)      | Verificável em `reports/quadro6.txt`         |
| Teste Diebold-Mariano α=5%              | `src/metrics.py`                             |
| Camada de coleta IoT (ESP32 → nuvem)    | `servidor/servidor_api.py` + `src/api_client.py` |
| Integração coleta → persistência        | `src/api_ingest.py` + `db/schema_api.sql`    |

## Observações

- **Dados**: o sistema gira em torno dos sensores. O banco guarda apenas o que
  o ESP32 mediu (mais os catálogos de referência e os cadastros feitos à mão).
  O dataset do Kaggle sobrevive fora do banco, lido do CSV, como base de
  comparação para o Quadro 6. Os vetores de entrada diferem — `pH, EC,
  temperature, air_temp` nos sensores e `pH, EC, temperature, humidity` no
  Kaggle, porque o protótipo mede duas temperaturas e não mede umidade — então
  cada fonte tem seus próprios artefatos, selecionados pelo parâmetro `fonte`
  em `model_lstm.run`, `model_arima.run` e `src.predict`.
- **Arquitetura de coleta**: o ESP32 não escreve no banco da aplicação. Ele
  publica em um serviço REST independente (FastAPI + SQLite em modo WAL), e a
  aplicação consome esse serviço por HTTP. O desacoplamento permite hospedar o
  servidor na nuvem e trocar o transporte (REST → MQTT) sem alterar o restante
  do sistema.
- **SGBD**: SQLite conforme seção 6.4 (portátil, sem servidor). O schema é
  compatível com PostgreSQL/MySQL para migração futura.
- **Modelos**: ARIMA usa refit expansivo em janela; LSTM tem 2 camadas
  recorrentes (64→32) + dropout, janela de 24 passos.
