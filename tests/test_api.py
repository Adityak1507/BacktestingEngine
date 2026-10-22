import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

client = TestClient(app)
SMALL = {"source": "synthetic", "days": 300, "seed": 1}


def test_health_and_strategies():
    assert client.get("/api/health").json()["status"] == "ok"
    ids = [s["id"] for s in client.get("/api/strategies").json()]
    assert ids == ["sma_crossover", "rsi_mean_reversion", "buy_and_hold"]


def test_backtest_returns_series_metrics_and_trades():
    body = {
        "data": SMALL,
        "strategy": {"name": "sma_crossover", "params": {"fast": 5, "slow": 20}},
        "costs": {"initial_cash": 50_000, "commission_bps": 10, "slippage_bps": 5},
    }
    r = client.post("/api/backtest", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["meta"]["bars"] == 300 and out["meta"]["symbol"] == "Synthetic"
    assert len(out["series"]) == 300
    first = out["series"][0]
    assert first["equity"] == 50_000 and first["drawdown"] == 0
    assert out["metrics"]["num_trades"] == len(out["trades"]) > 0
    assert out["fills"][0]["timestamp"].count("-") == 2  # ISO date string


def test_nan_metrics_serialise_as_null():
    body = {"data": SMALL, "strategy": {"name": "buy_and_hold"}}
    out = client.post("/api/backtest", json=body).json()
    assert out["metrics"]["win_rate"] is None  # no closed trades


def test_csv_source():
    csv = "Date,Open,High,Low,Close,Volume\n" + "\n".join(
        f"2024-01-{d:02d},{100+d},{101+d},{99+d},{100+d},1000" for d in range(1, 31)
    )
    body = {"data": {"source": "csv", "csv_text": csv, "csv_name": "test.csv"}, "strategy": {"name": "buy_and_hold"}}
    out = client.post("/api/backtest", json=body).json()
    assert out["meta"]["symbol"] == "test" and out["meta"]["bars"] == 30


def test_invalid_inputs_return_422():
    bad_sma = {"data": SMALL, "strategy": {"name": "sma_crossover", "params": {"fast": 50, "slow": 20}}}
    assert client.post("/api/backtest", json=bad_sma).status_code == 422
    assert client.post("/api/backtest", json={"data": {"source": "yahoo"}}).status_code == 422
    bad_csv = {"data": {"source": "csv", "csv_text": "date,close\n2024-01-01,2"}}
    r = client.post("/api/backtest", json=bad_csv)
    assert r.status_code == 422 and "missing columns" in r.json()["detail"]


def test_sweep_grid():
    body = {"data": SMALL, "fast_min": 5, "fast_max": 15, "slow_min": 10, "slow_max": 30, "step": 5}
    r = client.post("/api/sweep", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["fast_values"] == [5, 10, 15] and out["slow_values"] == [10, 15, 20, 25, 30]
    pairs = {(row["fast"], row["slow"]) for row in out["rows"]}
    assert all(f < s for f, s in pairs) and len(pairs) == out["meta"]["pairs"]


def test_sweep_rejects_huge_grid():
    body = {"data": SMALL, "fast_min": 2, "fast_max": 1000, "slow_min": 3, "slow_max": 1000, "step": 1}
    assert client.post("/api/sweep", json=body).status_code == 422


@pytest.mark.parametrize(
    "strategy",
    [
        {"name": "sma_crossover", "params": {"fast": 0, "slow": 50}},
        {"name": "sma_crossover", "params": {"fast": -5, "slow": 50}},
        {"name": "rsi_mean_reversion", "params": {"period": 0}},
        {"name": "rsi_mean_reversion", "params": {"oversold": 50, "exit_level": 40}},
        {"name": "sma_crossover", "params": {"fats": 10}},
    ],
)
def test_out_of_range_strategy_params_return_422(strategy):
    r = client.post("/api/backtest", json={"data": SMALL, "strategy": strategy})
    assert r.status_code == 422, r.text


def test_sweep_rejects_enormous_range_without_building_it():
    import time

    body = {"data": SMALL, "fast_min": 2, "fast_max": 10**12, "slow_min": 3, "slow_max": 4, "step": 1}
    started = time.perf_counter()
    assert client.post("/api/sweep", json=body).status_code == 422
    assert time.perf_counter() - started < 1
