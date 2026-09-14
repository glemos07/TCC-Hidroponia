-- Schema relacional do sistema de monitoramento hidropônico
-- TCC UNIP 2026 - Elber e Guilherme
-- 10 entidades conforme APÊNDICE A do documento

PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS monitoring_nutrient;
DROP TABLE IF EXISTS sensor_data;
DROP TABLE IF EXISTS farm;
DROP TABLE IF EXISTS monitoring;
DROP TABLE IF EXISTS nutrient;
DROP TABLE IF EXISTS sensor;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS ambient;
DROP TABLE IF EXISTS plants;
DROP TABLE IF EXISTS farm_status;

CREATE TABLE farm_status (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    status_name  VARCHAR(50) NOT NULL UNIQUE
);

CREATE TABLE plants (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name_plant   VARCHAR(100) NOT NULL,
    diversity    VARCHAR(100) NOT NULL,
    dt_planting  DATE NOT NULL,
    status_id    INTEGER NOT NULL,
    FOREIGN KEY (status_id) REFERENCES farm_status(id)
);

CREATE TABLE ambient (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    nm_ambient   VARCHAR(100) NOT NULL,
    tp_ambient   VARCHAR(50)  NOT NULL,
    descriptions VARCHAR(255)
);

CREATE TABLE users (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         VARCHAR(100) NOT NULL,
    email        VARCHAR(150) NOT NULL UNIQUE,
    role         VARCHAR(50)  NOT NULL
);

CREATE TABLE sensor (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    sensor_type  VARCHAR(50)  NOT NULL UNIQUE,
    unit         VARCHAR(20)  NOT NULL,
    description  VARCHAR(255)
);

CREATE TABLE monitoring (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    dt_register  DATETIME NOT NULL,
    id_plant     INTEGER NOT NULL,
    id_ambient   INTEGER NOT NULL,
    id_user      INTEGER,
    obs          VARCHAR(255),
    FOREIGN KEY (id_plant)   REFERENCES plants(id),
    FOREIGN KEY (id_ambient) REFERENCES ambient(id),
    FOREIGN KEY (id_user)    REFERENCES users(id)
);

CREATE TABLE sensor_data (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    id_monitoring INTEGER NOT NULL,
    id_sensor     INTEGER NOT NULL,
    value         REAL    NOT NULL,
    FOREIGN KEY (id_monitoring) REFERENCES monitoring(id),
    FOREIGN KEY (id_sensor)     REFERENCES sensor(id)
);

CREATE TABLE nutrient (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         VARCHAR(100) NOT NULL UNIQUE
);

CREATE TABLE monitoring_nutrient (
    id_monitoring INTEGER NOT NULL,
    id_nutrient   INTEGER NOT NULL,
    PRIMARY KEY (id_monitoring, id_nutrient),
    FOREIGN KEY (id_monitoring) REFERENCES monitoring(id),
    FOREIGN KEY (id_nutrient)   REFERENCES nutrient(id)
);

CREATE TABLE farm (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    dt_harvest     DATE    NOT NULL,
    weight_harvest REAL    NOT NULL,
    time_growth    REAL    NOT NULL,
    obs            VARCHAR(255),
    id_plant       INTEGER NOT NULL,
    id_ambient     INTEGER NOT NULL,
    FOREIGN KEY (id_plant)   REFERENCES plants(id),
    FOREIGN KEY (id_ambient) REFERENCES ambient(id)
);

CREATE INDEX idx_monitoring_plant   ON monitoring(id_plant);
CREATE INDEX idx_monitoring_ambient ON monitoring(id_ambient);
CREATE INDEX idx_monitoring_date    ON monitoring(dt_register);
CREATE INDEX idx_sensordata_mon     ON sensor_data(id_monitoring);
CREATE INDEX idx_sensordata_sensor  ON sensor_data(id_sensor);
CREATE INDEX idx_farm_plant         ON farm(id_plant);
