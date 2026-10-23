"""Machine-learning direction classifier with leakage-free walk-forward evaluation.

The classifier predicts whether the *tradeable* next-day return is positive:
a decision made at bar t's close fills at bar t+1's open, so the label is

    y[t] = 1  if  open[t+2] > open[t+1]  else 0

Features use only data available at bar t's close. Models are evaluated
walk-forward: retrained every `step` bars on all labels already known, then
used to predict the following block. A purge gap keeps training labels from
overlapping the first test bar, so every prediction is out of sample.

Requires scikit-learn (`pip install -e ".[ml]"`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .data import validate_ohlcv
from .strategy import Context, Strategy


def purge_gap(horizon: int) -> int:
    """A label at bar s needs open[s+1+horizon], so at bar t only labels with
    s <= t - (1 + horizon) are known. Training rows closer than this are dropped."""
    return 1 + horizon


# --- features & labels -----------------------------------------------------
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Technical features at each bar, using only data up to that bar's close."""
    df = validate_ohlcv(df)
    o, h, l, c, v = (df[k] for k in ("open", "high", "low", "close", "volume"))
    logret = np.log(c).diff()
    f = pd.DataFrame(index=df.index)

    for n in (1, 2, 5, 10, 20):
        f[f"ret_{n}"] = np.log(c / c.shift(n))
    f["vol_5"] = logret.rolling(5).std()
    f["vol_20"] = logret.rolling(20).std()
    f["vol_ratio"] = f["vol_5"] / f["vol_20"]
    f["rsi_14"] = _rsi(c, 14) / 100
    for n in (10, 20, 50):
        f[f"dist_sma_{n}"] = c / c.rolling(n).mean() - 1
    f["sma_10_50"] = c.rolling(10).mean() / c.rolling(50).mean() - 1
    f["bb_z_20"] = (c - c.rolling(20).mean()) / c.rolling(20).std()
    f["range_hl"] = (h - l) / c
    f["close_loc"] = ((c - l) / (h - l)).where(h > l, 0.5)
    f["gap"] = o / c.shift(1) - 1
    logv = np.log(v.replace(0, np.nan))
    f["volume_z_20"] = (logv - logv.rolling(20).mean()) / logv.rolling(20).std()
    return f.replace([np.inf, -np.inf], np.nan)


def make_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Tradeable forward return and its direction.

    `fwd[t]` = open[t+1+horizon] / open[t+1] - 1 (enter at the next open, hold
    `horizon` bars); `y[t]` = 1 if `fwd[t]` > 0. NaN where the future is unknown.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    df = validate_ohlcv(df)
    fwd = df["open"].shift(-1 - horizon) / df["open"].shift(-1) - 1
    y = (fwd > 0).astype(float).where(fwd.notna())
    return pd.DataFrame({"y": y, "fwd": fwd})


# --- models ----------------------------------------------------------------
def _model_factories() -> Dict[str, Callable[[], object]]:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=40, max_features="sqrt",
            n_jobs=-1, random_state=0,
        ),
        "gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.03, max_iter=150, min_samples_leaf=60,
            l2_regularization=1.0, random_state=0,
        ),
    }


MODELS = ("logreg", "random_forest", "gradient_boosting")
BASELINES = ("always_up", "momentum")


class _Baseline:
    """Non-learning reference predictors with a sklearn-like interface."""

    def __init__(self, kind: str):
        self.kind = kind
        self.up_rate = 0.5

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.up_rate = float(np.mean(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.kind == "always_up":
            p = np.full(len(X), max(self.up_rate, 0.5 + 1e-9))
        else:  # momentum: yesterday's direction persists
            p = np.where(X["ret_1"].to_numpy() > 0, 0.55, 0.45)
        return np.column_stack([1 - p, p])


def _make(name: str):
    if name in BASELINES:
        return _Baseline(name)
    factories = _model_factories()
    if name not in factories:
        raise ValueError(f"unknown model {name!r}; choose from {MODELS + BASELINES}")
    return factories[name]()


# --- metrics ---------------------------------------------------------------
def classification_metrics(y: np.ndarray, pred: np.ndarray, proba: Optional[np.ndarray] = None,
                           fwd: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Accuracy, precision, recall, F1, ROC AUC and the trading edge of the 'up' calls."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    out = {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)) if pred.any() else float("nan"),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "coverage": float(pred.mean()),  # share of days predicted 'up'
        "roc_auc": float("nan"),
    }
    if proba is not None and len(np.unique(y)) == 2 and np.ptp(proba) > 0:
        out["roc_auc"] = float(roc_auc_score(y, proba))
    if fwd is not None:
        fwd = np.asarray(fwd)
        out["avg_fwd_all"] = float(fwd.mean())
        out["avg_fwd_predicted_up"] = float(fwd[pred == 1].mean()) if pred.any() else float("nan")
    return out


def pick_threshold(y: np.ndarray, proba: np.ndarray, min_recall: float = 0.3) -> float:
    """Highest-precision probability cut-off that still keeps recall >= `min_recall`.

    Chosen on validation data only, never on the test period.
    """
    y = np.asarray(y).astype(int)
    if y.sum() == 0:
        return 0.5
    best_t, best_p = 0.5, -1.0
    for t in np.unique(np.quantile(proba, np.linspace(0.0, 0.9, 46))):
        pred = proba >= t
        if not pred.any():
            continue
        recall = (pred & (y == 1)).sum() / y.sum()
        if recall < min_recall:
            continue
        precision = (pred & (y == 1)).sum() / pred.sum()
        if precision > best_p:
            best_t, best_p = float(t), float(precision)
    return best_t


# --- walk-forward ----------------------------------------------------------
RULES = ("default", "tuned", "confident")


@dataclass
class WalkForwardResult:
    model: str
    horizon: int
    predictions: pd.DataFrame  # out-of-sample rows: proba, thr_*, pred_*, y, fwd
    feature_importance: Optional[pd.Series] = None
    n_refits: int = 0

    def metrics(self, rule: str = "default", start=None, end=None) -> Dict[str, float]:
        """Metrics for one decision rule, optionally restricted to a date range."""
        p = self.predictions.loc[start:end]
        return classification_metrics(p["y"], p[f"pred_{rule}"], p["proba"], p["fwd"])

    def signal(self, rule: str = "default") -> pd.Series:
        """Boolean 'be long' signal per date, usable by `MLSignalStrategy`."""
        return self.predictions[f"pred_{rule}"].astype(bool)


def walk_forward(
    df: pd.DataFrame,
    model: str = "logreg",
    horizon: int = 1,
    min_train: int = 756,
    step: int = 63,
    val_size: int = 252,
    min_recall: float = 0.3,
    confidence: float = 0.3,
    features: Optional[List[str]] = None,
) -> WalkForwardResult:
    """Out-of-sample predictions for every bar after the first `min_train` usable bars.

    At each refit bar t0 the model trains only on rows whose labels are already
    known (s <= t0 - purge_gap(horizon)). Three decision rules are recorded:

    - ``default``: predict 'up' when probability >= 0.5.
    - ``tuned``: the cut-off with the best precision (recall >= `min_recall`)
      on a validation slice, the last `val_size` known rows.
    - ``confident``: predict 'up' only for the most confident `confidence`
      share of days, using the validation slice's probability quantile.

    Both data-driven cut-offs come from a model fitted *before* the validation
    slice; the model is then refitted on all known rows to predict the next
    `step` bars.
    """
    purge = purge_gap(horizon)
    X_all = make_features(df)
    if features:
        X_all = X_all[features]
    tgt = make_target(df, horizon)
    usable = X_all.notna().all(axis=1).to_numpy()
    positions = np.flatnonzero(usable)
    if len(positions) <= min_train + purge:
        raise ValueError("not enough data for walk-forward evaluation")

    y_all = tgt["y"].to_numpy()
    rows = []
    importances = []
    n_refits = 0
    for t0 in range(positions[min_train], positions[-1] + 1, step):
        train_pos = positions[(positions <= t0 - purge) & ~np.isnan(y_all[positions])]
        test_pos = positions[(positions >= t0) & (positions < t0 + step)]
        if len(test_pos) == 0 or len(train_pos) < 2 * val_size:
            continue

        # 1) cut-offs from a validation slice the fitted model has not seen
        fit_pos, val_pos = train_pos[: -val_size - purge], train_pos[-val_size:]
        m = _make(model).fit(X_all.iloc[fit_pos], y_all[fit_pos])
        val_proba = m.predict_proba(X_all.iloc[val_pos])[:, 1]
        thr_tuned = pick_threshold(y_all[val_pos], val_proba, min_recall)
        thr_conf = float(np.quantile(val_proba, 1 - confidence))

        # 2) refit on everything known and predict the next block
        m = _make(model).fit(X_all.iloc[train_pos], y_all[train_pos])
        proba = m.predict_proba(X_all.iloc[test_pos])[:, 1]
        n_refits += 1
        imp = getattr(m, "feature_importances_", None)
        if imp is None and hasattr(m, "named_steps"):
            imp = np.abs(m.named_steps["logisticregression"].coef_[0])
        if imp is not None:
            importances.append(np.asarray(imp) / max(np.sum(imp), 1e-12))

        for pos, pr in zip(test_pos, proba):
            rows.append((X_all.index[pos], pr, 0.5, thr_tuned, thr_conf))

    pred = pd.DataFrame(rows, columns=["date", "proba", "thr_default", "thr_tuned", "thr_confident"])
    pred = pred.set_index("date")
    for rule in RULES:
        pred[f"pred_{rule}"] = (pred["proba"] >= pred[f"thr_{rule}"]).astype(int)
    pred = pred.join(tgt)
    pred = pred[pred["y"].notna()]
    importance = (
        pd.Series(np.mean(importances, axis=0), index=X_all.columns).sort_values(ascending=False)
        if importances else None
    )
    return WalkForwardResult(model, horizon, pred, importance, n_refits)


# --- trading ---------------------------------------------------------------
class MLSignalStrategy(Strategy):
    """Long while the classifier's out-of-sample signal says 'up', flat otherwise.

    `signal` is a boolean Series indexed by date (e.g. `WalkForwardResult.signal()`),
    known at each bar's close. Bars without a signal are ignored.
    """

    def __init__(self, signal: pd.Series, fraction: float = 0.99, symbol: Optional[str] = None):
        self.signal = signal.astype(bool)
        self.fraction, self.symbol = fraction, symbol

    def on_bar(self, ctx: Context) -> None:
        if ctx.timestamp not in self.signal.index:
            return
        long = bool(self.signal.loc[ctx.timestamp])
        pos = ctx.position(self.symbol)
        if long and pos == 0:
            ctx.order_target_percent(self.fraction, self.symbol)
        elif not long and pos > 0:
            ctx.close_position(self.symbol)
