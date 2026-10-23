import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from backtester import Backtest, generate_gbm  # noqa: E402
from backtester import ml  # noqa: E402
from backtester.data import adjust_prices  # noqa: E402
from backtester.ml import (  # noqa: E402
    MLSignalStrategy,
    classification_metrics,
    make_features,
    make_target,
    pick_threshold,
    purge_gap,
    walk_forward,
)


def predictable_data(n=1500, seed=0):
    """Prices whose next open-to-open move follows today's close location in the bar.

    A model that learns `close_loc` can predict direction; a model that leaks
    nothing still has to learn it from the past.
    """
    rng = np.random.default_rng(seed)
    loc = rng.uniform(0, 1, n)
    opens = [100.0]
    for t in range(n - 1):
        drift = 0.01 if loc[t] > 0.5 else -0.01
        opens.append(opens[-1] * (1 + drift + rng.normal(0, 0.004)))
    opens = np.array(opens)
    # Shift so open[t+1] -> open[t+2] depends on loc[t]: decision at t, trade from t+1.
    opens = np.concatenate([[100.0], opens[:-1]])
    width = rng.uniform(0.005, 0.03, n)  # random bar size, so only close_loc carries the signal
    high = opens * (1 + width)
    low = opens * (1 - width)
    close = low + loc * (high - low)
    idx = pd.bdate_range("2000-01-03", periods=n)
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    return pd.DataFrame({"open": opens, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


# --- no look-ahead -----------------------------------------------------------
def test_features_do_not_use_future_bars():
    data = generate_gbm(400, seed=3)
    full = make_features(data)
    truncated = make_features(data.iloc[:250])
    pd.testing.assert_frame_equal(full.iloc[:250], truncated)


def test_target_is_next_open_to_following_open():
    data = generate_gbm(50, seed=1)
    tgt = make_target(data)
    o = data["open"].to_numpy()
    assert tgt["fwd"].iloc[10] == pytest.approx(o[12] / o[11] - 1)
    assert tgt["y"].iloc[10] == float(o[12] > o[11])
    assert tgt["y"].iloc[-2:].isna().all()

    tgt5 = make_target(data, horizon=5)
    assert tgt5["fwd"].iloc[10] == pytest.approx(o[16] / o[11] - 1)
    assert tgt5["y"].iloc[-6:].isna().all()


def test_walk_forward_only_trains_on_known_labels(monkeypatch):
    data = generate_gbm(1300, seed=2)
    seen = []

    class Spy:
        def fit(self, X, y):
            self.last_train = X.index.max()
            return self

        def predict_proba(self, X):
            seen.append((self.last_train, X.index.min()))
            return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])

    monkeypatch.setattr(ml, "_make", lambda name: Spy())
    walk_forward(data, model="anything", min_train=600, step=50, val_size=150, horizon=3)

    pos = {ts: i for i, ts in enumerate(data.index)}
    gaps = [pos[first_pred] - pos[last_train] for last_train, first_pred in seen]
    assert gaps and min(gaps) >= purge_gap(3)


def test_walk_forward_learns_a_real_pattern():
    data = predictable_data()
    res = walk_forward(data, model="logreg", min_train=500, step=50, val_size=150)
    m = res.metrics()
    assert m["roc_auc"] > 0.9 and m["accuracy"] > 0.85
    assert res.feature_importance.index[0] == "close_loc"


def test_walk_forward_finds_nothing_in_a_random_walk():
    data = generate_gbm(2000, seed=7)
    m = walk_forward(data, model="logreg", min_train=600, step=63, val_size=200).metrics()
    assert 0.44 < m["roc_auc"] < 0.56


# --- metrics and thresholds -------------------------------------------------
def test_classification_metrics_known_values():
    y = np.array([1, 1, 0, 0, 1])
    pred = np.array([1, 0, 1, 0, 1])
    m = classification_metrics(y, pred, proba=np.array([0.9, 0.4, 0.6, 0.2, 0.8]), fwd=np.array([1, 2, -1, -2, 3.0]))
    assert m["accuracy"] == pytest.approx(0.6)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["coverage"] == pytest.approx(0.6)
    assert m["avg_fwd_predicted_up"] == pytest.approx(1.0)


def test_pick_threshold_trades_recall_for_precision():
    rng = np.random.default_rng(0)
    proba = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < proba).astype(int)  # higher proba -> more likely up
    loose = pick_threshold(y, proba, min_recall=0.9)
    strict = pick_threshold(y, proba, min_recall=0.3)
    assert strict > loose


def test_confident_rule_predicts_fewer_ups():
    data = predictable_data(1200, seed=4)
    res = walk_forward(data, model="logreg", min_train=500, step=50, val_size=150, confidence=0.2)
    assert res.metrics("confident")["coverage"] < res.metrics("default")["coverage"]


# --- trading and data ------------------------------------------------------
def test_ml_signal_strategy_follows_signal():
    data = generate_gbm(30, seed=1)
    signal = pd.Series(False, index=data.index)
    signal.iloc[5:10] = True
    result = Backtest(data, MLSignalStrategy(signal)).run()
    fills = result.fills
    assert len(fills) == 2
    assert fills.iloc[0]["timestamp"] == data.index[6] and fills.iloc[0]["quantity"] > 0
    assert fills.iloc[1]["timestamp"] == data.index[11] and fills.iloc[1]["quantity"] < 0


def test_adjust_prices_uses_adj_close_ratio():
    raw = pd.DataFrame(
        {"Open": [10.0, 20.0], "High": [11.0, 22.0], "Low": [9.0, 18.0], "Close": [10.0, 20.0],
         "Adj Close": [5.0, 20.0], "Volume": [1, 1]},
        index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
    )
    adj = adjust_prices(raw)
    assert adj["close"].tolist() == [5.0, 20.0]
    assert adj["open"].tolist() == [5.0, 20.0]
    assert "adj close" not in adj.columns
