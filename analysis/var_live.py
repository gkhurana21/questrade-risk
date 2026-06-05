"""
VaR for live positions — reuses the validated Phase 3 methodology.

Pulls historical daily returns for the underlying equities via yfinance
(network already confirmed for Yahoo), builds a portfolio return series
weighted by current delta exposure, and runs the same 252-day rolling
historical-simulation VaR that passed the Phase 3 backtest.

This is an observation module: it reports VaR as a risk measure.
It does not suggest any action.
"""

import logging
from typing import List, Optional
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from .pricer import PortfolioSnapshot, PositionGreeks

log = logging.getLogger(__name__)

LOOKBACK = 252   # trading days


def _fetch_returns(tickers: List[str], lookback: int = LOOKBACK + 5) -> pd.DataFrame:
    """Pull daily adjusted returns for a list of tickers via yfinance."""
    try:
        import yfinance as yf
        raw = yf.download(tickers, period=f"{lookback + 20}d",
                          auto_adjust=True, progress=False)["Close"]
        if isinstance(raw, pd.Series):
            raw = raw.to_frame(tickers[0])
        return np.log(raw / raw.shift(1)).dropna()
    except Exception as exc:
        log.warning("yfinance fetch failed: %s", exc)
        return pd.DataFrame()


def compute_var(snap: PortfolioSnapshot) -> None:
    """
    Compute 1-day 95% portfolio VaR and write it into snap in-place.

    Weight each underlying by its current net delta-dollar exposure:
      weight_i = Σ(position.delta × qty × multiplier × spot)  for all positions
                  on underlying i, normalised to sum = 1.
    """
    # Collect underlying tickers and their net delta-dollar weight
    exposure: dict = {}   # ticker → delta-dollar exposure
    for p in snap.positions:
        if p.error or not p.is_option:
            continue
        und = p.underlying or p.symbol.split(" ")[0]
        mult = 100
        doll = p.delta * p.qty * mult * (p.mkt_price or 0)
        exposure[und] = exposure.get(und, 0.0) + doll

    if not exposure:
        log.info("No option positions with valid Greeks — skipping VaR")
        return

    tickers = [t for t in exposure if exposure[t] != 0]
    if not tickers:
        return

    log.info("Fetching historical returns for VaR: %s", tickers)
    rets = _fetch_returns(tickers)
    if rets.empty or len(rets) < LOOKBACK:
        log.warning("Insufficient return history (%d rows) for VaR", len(rets))
        return

    # Normalise weights
    total_exp = sum(abs(v) for v in exposure.values())
    if total_exp == 0:
        return
    weights = np.array([exposure.get(t, 0.0) / total_exp for t in tickers])

    # Build weighted portfolio returns (last LOOKBACK rows)
    avail = [t for t in tickers if t in rets.columns]
    if not avail:
        return
    w_avail = np.array([exposure.get(t, 0.0) for t in avail])
    w_avail = w_avail / (w_avail.sum() or 1.0)

    port_ret = (rets[avail].iloc[-LOOKBACK:] * w_avail).sum(axis=1)
    arr = port_ret.values

    # Historical VaR
    var_h = -np.percentile(arr, 5.0)

    # Parametric VaR (normal approximation)
    mu, sig = arr.mean(), arr.std(ddof=1)
    var_p = -(mu - 1.6449 * sig)

    # Express as dollar P&L on notional
    snap.var_hist_1d  = var_h  * total_exp
    snap.var_param_1d = var_p  * total_exp

    log.info("VaR computed: hist=%.2f  param=%.2f  on notional %.0f",
             snap.var_hist_1d, snap.var_param_1d, total_exp)
