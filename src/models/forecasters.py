"""Model peramalan per negara dan backtest rolling-origin dengan metrik SMAPE."""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")
Forecaster = Callable[[pd.Series, int, dict], np.ndarray]


def smape(actual, forecast) -> float:
    """SMAPE dalam persen (0–200)."""
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    return float(100 * np.mean(2 * np.abs(f - a) / (np.abs(a) + np.abs(f) + 1e-9)))


def log_growth(s: pd.Series, years: int) -> float:
    """Kemiringan tren log per bulan pada `years` tahun terakhir (regresi linear)."""
    x = np.log(s.dropna().iloc[-years * 12:])
    return float(np.polyfit(np.arange(len(x)), x.values, 1)[0])


def _from_level(s: pd.Series, h: int, slope: float) -> np.ndarray:
    """Proyeksi dari rata-rata log 12 bulan terakhir (titik tengahnya 5,5 bulan sebelum akhir)."""
    level = np.log(s.dropna().iloc[-12:]).mean()
    return np.exp(level + slope * (np.arange(1, h + 1) + 5.5))


def fc_last(s, h, ctx=None):
    return np.repeat(s.dropna().iloc[-1], h)


def fc_mean12(s, h, ctx=None):
    return np.repeat(s.dropna().iloc[-12:].mean(), h)


def make_drift(years: int) -> Forecaster:
    def fc(s, h, ctx=None):
        return _from_level(s, h, log_growth(s, years))
    fc.__name__ = f"drift_{years}y"
    return fc


def fc_damped(s, h, ctx=None):
    x = np.log(s.dropna()).iloc[-15 * 12:]
    return np.exp(ExponentialSmoothing(x.values, trend="add", damped_trend=True).fit().forecast(h))


def fc_damped_seasonal(s, h, ctx=None):
    x = np.log(s.dropna()).iloc[-15 * 12:]
    m = ExponentialSmoothing(x.values, trend="add", damped_trend=True, seasonal="add", seasonal_periods=12).fit()
    return np.exp(m.forecast(h))


def make_pooled_drift(weight: float, years: int = 5) -> Forecaster:
    """Pertumbuhan negara disusutkan ke median pertumbuhan semua negara: (1-w)·g_negara + w·g_median."""
    def fc(s, h, ctx):
        g = (1 - weight) * log_growth(s, years) + weight * ctx["median_growth"]
        return _from_level(s, h, g)
    fc.__name__ = f"pooled_drift_w{weight:.2f}"
    return fc


def make_ensemble(*members: Forecaster) -> Forecaster:
    def fc(s, h, ctx):
        return np.mean([m(s, h, ctx) for m in members], axis=0)
    fc.__name__ = "ensemble(" + " + ".join(m.__name__ for m in members) + ")"
    return fc


def context(panel: pd.DataFrame, cut: int, min_months: int = 180, years: int = 5) -> dict:
    """Informasi lintas negara yang hanya memakai data sebelum `cut` (tanpa kebocoran)."""
    hist = {c: panel[c][panel.index < cut] for c in panel}
    hist = {c: s for c, s in hist.items() if s.dropna().shape[0] >= min_months}
    return {"hist": hist, "median_growth": float(np.median([log_growth(s, years) for s in hist.values()]))}


def backtest(panel_clean: pd.DataFrame, panel_raw: pd.DataFrame, models: dict[str, Forecaster],
             origins: list[int], horizon: int = 120) -> pd.DataFrame:
    """Untuk setiap origin (tahun terakhir data latih), ramal `horizon` bulan dan ukur SMAPE per negara.

    Target dievaluasi pada data mentah (NaN dan 0 diabaikan), sama seperti nilai yang tersedia di leaderboard.
    """
    rows = []
    for origin in origins:
        cut = (origin + 1) * 12
        ctx = context(panel_clean, cut)
        for c, s in ctx["hist"].items():
            actual = panel_raw[c][(panel_raw.index >= cut) & (panel_raw.index < cut + horizon)].replace(0, np.nan)
            for name, f in models.items():
                fc = pd.Series(f(s, horizon, ctx), index=range(cut, cut + horizon))
                ok = actual.notna()
                err = 2 * np.abs(fc[ok] - actual[ok]) / (np.abs(actual[ok]) + np.abs(fc[ok]))
                rows.append({"origin": origin, "country": c, "model": name, "smape": 100 * err.mean(),
                             "smape_by_year": (100 * err.groupby((err.index - cut) // 12).mean()).to_dict()})
    return pd.DataFrame(rows)
