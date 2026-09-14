# Dataset do Kaggle

Coloque aqui o(s) arquivo(s) CSV baixado(s) de:
https://www.kaggle.com/datasets/itsmonir31/hydroponics-datasets/data

## Como usar

1. Baixe o CSV do Kaggle (é preciso ter conta gratuita).
2. Extraia o(s) arquivo(s) `.csv` para esta pasta.
3. Do diretório `app/`, execute:

```powershell
& "$env:USERPROFILE\miniconda3\envs\tcc-hidroponia\python.exe" -m src.seed
```

O seed detectará automaticamente o CSV, aplicará limpeza (nulos, duplicatas)
conforme Quadro 1 do documento e populará o banco.

## Colunas reconhecidas

O loader detecta os seguintes campos por sinônimos (case-insensitive):

| Alvo         | Nomes aceitos                                                        |
|--------------|----------------------------------------------------------------------|
| timestamp    | timestamp, datetime, date_time, date, time, dt, reading_time         |
| pH           | ph, p_h, ph_value, ph_level                                          |
| EC           | ec, conductivity, electrical_conductivity, ec_ms, ec_mscm, tds       |
| temperature  | temperature, temp, water_temp, temp_c, ambient_temperature           |
| humidity     | humidity, hum, rh, relative_humidity, moisture                       |
| plant_id     | plant_id, plant, crop_id, planta                                     |
| growth_stage | growth_stage, stage, phase, growth                                   |
| yield        | yield, harvest_weight, weight, production                            |

Se algum campo essencial (pH, EC, temperature, humidity) não existir no CSV,
ele fica NaN. Se timestamp faltar, é sintetizado com intervalo de 15 min.

Se **nenhum** CSV estiver aqui, o pipeline usa o gerador sintético reprodutível.
