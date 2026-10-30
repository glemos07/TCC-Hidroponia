"""
Compara ARIMA vs LSTM e produz o Quadro 6 do documento
(desempenho comparativo dos modelos preditivos), o Gráfico 3
(observado vs previsto) e o teste Diebold-Mariano.

Uso:
    python -m src.evaluate
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ARTIFACT_SUFFIX, REPORTS_DIR
from src.metrics import rmse, mae, mape, diebold_mariano
from src import model_arima, model_lstm


def _align(a: list, b: list) -> tuple[list, list]:
    n = min(len(a), len(b))
    return a[-n:], b[-n:]


def evaluate_target(target: str, fonte: str = "kaggle") -> dict:
    arima_res = model_arima.run(target=target, fonte=fonte)
    lstm_res  = model_lstm.run(target=target, fonte=fonte)

    y_true_a, y_pred_a = arima_res["y_true"], arima_res["y_pred"]
    y_true_l, y_pred_l = lstm_res["y_true"],  lstm_res["y_pred"]

    # alinha comprimentos para comparação DM (LSTM tem janela inicial)
    y_true_a, y_pred_a = _align(y_true_a, y_pred_a)
    y_true_l, y_pred_l = _align(y_true_l, y_pred_l)
    n = min(len(y_true_a), len(y_true_l))
    y_true = y_true_l[-n:]
    y_a = y_pred_a[-n:]
    y_l = y_pred_l[-n:]

    metrics = {
        "target": target,
        "fonte": fonte,
        "ARIMA": {
            "RMSE": rmse(y_true, y_a),
            "MAE":  mae(y_true, y_a),
            "MAPE": mape(y_true, y_a),
        },
        "LSTM": {
            "RMSE": rmse(y_true, y_l),
            "MAE":  mae(y_true, y_l),
            "MAPE": mape(y_true, y_l),
        },
    }
    dm_stat, dm_p = diebold_mariano(y_true, y_a, y_l, h=1, loss="squared")
    metrics["diebold_mariano"] = {"stat": dm_stat, "p_value": dm_p}

    # Gráfico 3 - observado vs previsto (LSTM)
    plot_dir = REPORTS_DIR
    plot_dir.mkdir(exist_ok=True, parents=True)
    plt.figure(figsize=(10, 4.5))
    plt.plot(y_true, label="Observado", color="#1f4e79", linewidth=2)
    plt.plot(y_l, label="Previsto (LSTM)", color="#c00000", linewidth=1.5, alpha=0.9)
    plt.plot(y_a, label="Previsto (ARIMA)", color="#7f7f7f", linewidth=1.2, linestyle="--", alpha=0.8)
    plt.title(f"Previsao de {target}: observado x previsto")
    plt.xlabel("Passos de teste")
    plt.ylabel(target)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / f"grafico3_{target}{ARTIFACT_SUFFIX[fonte]}.png", dpi=140)
    plt.close()

    return metrics


def format_quadro6(all_metrics: list[dict]) -> str:
    lines = []
    for m in all_metrics:
        tgt = m["target"]
        a = m["ARIMA"]; l = m["LSTM"]
        red_rmse = (l["RMSE"] - a["RMSE"]) / a["RMSE"] * 100
        red_mae  = (l["MAE"]  - a["MAE"])  / a["MAE"]  * 100
        red_mape = (l["MAPE"] - a["MAPE"]) / a["MAPE"] * 100
        lines.append(f"\nQuadro 6 - Desempenho comparativo dos modelos preditivos ({tgt})")
        lines.append("-" * 72)
        lines.append(f"{'Modelo':30s} {'RMSE':>10s} {'MAE':>10s} {'MAPE (%)':>10s}")
        lines.append(f"{'ARIMA (univariado)':30s} {a['RMSE']:>10.4f} {a['MAE']:>10.4f} {a['MAPE']:>10.2f}")
        lines.append(f"{'LSTM (multivariado)':30s} {l['RMSE']:>10.4f} {l['MAE']:>10.4f} {l['MAPE']:>10.2f}")
        # sinal negativo = LSTM melhor. Rotular como "reducao" so vale quando
        # o LSTM vence; com o ARIMA na frente o numero fica positivo e o rotulo
        # antigo se lia ao contrario.
        lines.append(f"{'Variacao LSTM vs ARIMA (- = LSTM melhor)':40s} "
                     f"{red_rmse:>+8.1f}% {red_mae:>+8.1f}% {red_mape:>+8.1f}%")
        vencedor = "LSTM" if l["RMSE"] < a["RMSE"] else "ARIMA"
        lines.append(f"{'Menor RMSE':40s} {vencedor}")
        dm = m["diebold_mariano"]
        sig = "significativa (rejeita H0)" if dm["p_value"] < 0.05 else "nao significativa"
        lines.append(f"Diebold-Mariano: stat={dm['stat']:.3f}  p-value={dm['p_value']:.4f}  ->  diferenca {sig}")
    return "\n".join(lines)


def main(fonte: str = "kaggle"):
    sufixo = ARTIFACT_SUFFIX[fonte]
    results = []
    for tgt in ["pH", "EC"]:
        results.append(evaluate_target(tgt, fonte=fonte))

    report = format_quadro6(results)
    print(report)

    (REPORTS_DIR / f"quadro6{sufixo}.txt").write_text(report, encoding="utf-8")
    with open(REPORTS_DIR / f"metrics{sufixo}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nArtefatos salvos em: {REPORTS_DIR}")


if __name__ == "__main__":
    import sys

    # `python -m src.evaluate api` avalia a telemetria real do ESP32
    main(sys.argv[1] if len(sys.argv) > 1 else "kaggle")
