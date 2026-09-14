"""Testes rapidos das metricas."""
import numpy as np
import pytest

from src.metrics import rmse, mae, mape, diebold_mariano


def test_rmse_zero_para_perfeito():
    y = [1.0, 2.0, 3.0]
    assert rmse(y, y) == pytest.approx(0.0)


def test_mae_conhecido():
    assert mae([1, 2, 3], [2, 3, 4]) == pytest.approx(1.0)


def test_mape_conhecido():
    assert mape([100, 200], [110, 180]) == pytest.approx(10.0, rel=0.02)


def test_diebold_mariano_retorna_valores():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 100)
    e_a = y + rng.normal(0, 1, 100)
    e_b = y + rng.normal(0, 0.5, 100)
    stat, p = diebold_mariano(y, e_a, e_b, h=1)
    assert isinstance(stat, float)
    assert 0 <= p <= 1
