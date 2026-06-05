"""
Hypothetical position analyser.

Takes a manually-specified contract (underlying, type, strike, expiry, qty)
and runs it through the SAME live-data path as real positions:
  1. Fetch live underlying spot from Questrade (yfinance fallback)
  2. Fetch real option chain → locate the contract → get market bid/ask mid
  3. Invert IV from that mid via the C++ engine (Newton-Raphson bs_full)
  4. Compute full Greeks via C++ bs_full
  5. Return a PositionGreeks labelled HYPOTHETICAL

This is a what-if calculator on real market data. It has no execution path.
"""

import math
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

RISK_FREE = 0.045   # same constant used throughout the live analysis


@dataclass
class HypotheticalSpec:
    underlying: str      # e.g. "AAPL"
    option_type: str     # "call" or "put"
    strike: float
    expiry: str          # "YYYY-MM-DD"
    qty: int             # + long, - short

    def label(self) -> str:
        return (f"{self.underlying} {self.option_type.upper()} "
                f"K={self.strike:.0f} exp={self.expiry} qty={self.qty:+d}  "
                f"[HYPOTHETICAL — not a real holding]")


@dataclass
class HypotheticalResult:
    spec:         HypotheticalSpec
    spot:         Optional[float]
    spot_source:  str                   # "questrade" | "yfinance" | "unavailable"
    market_bid:   Optional[float]
    market_ask:   Optional[float]
    market_mid:   Optional[float]
    mid_source:   str                   # "questrade" | "yfinance" | "unavailable"
    T_years:      Optional[float]
    implied_vol:  Optional[float]
    model_price:  Optional[float]
    delta:        Optional[float]
    gamma:        Optional[float]
    theta_day:    Optional[float]
    vega:         Optional[float]
    error:        Optional[str] = None

    def sanity_flags(self) -> list:
        """Return list of sanity warnings; empty = all clear."""
        flags = []
        if self.delta is None:
            return ["could not compute Greeks"]
        ot = self.spec.option_type.lower()
        if ot == "call" and not (0.0 <= self.delta <= 1.0):
            flags.append(f"call delta {self.delta:.4f} out of range [0,1]")
        if ot == "put" and not (-1.0 <= self.delta <= 0.0):
            flags.append(f"put delta {self.delta:.4f} out of range [-1,0]")
        if self.implied_vol is not None and not (0.01 <= self.implied_vol <= 3.0):
            flags.append(f"IV {self.implied_vol*100:.1f}% looks unreasonable")
        if self.gamma is not None and self.gamma < 0:
            flags.append(f"gamma {self.gamma:.6f} is negative (should be ≥0)")
        if self.T_years is not None and self.T_years < 0:
            flags.append("expiry is in the past")
        return flags


# ── live data fetch ────────────────────────────────────────────────────────────

def _spot_from_questrade(client, underlying: str) -> tuple:
    """Returns (spot, symbol_id) or (None, None)."""
    try:
        results = client.symbol_search(underlying)
        # pick the exact equity match (not options/warrants)
        equity = next(
            (s for s in results
             if s.get("symbol", "").upper() == underlying.upper()
             and s.get("securityType", "") in ("Stock", "Equity", "ETF")),
            None
        )
        if not equity:
            equity = results[0] if results else None
        if not equity:
            return None, None
        sym_id = equity["symbolId"]
        q = client.quote(sym_id)
        spot = float(
            q.get("lastTradePriceTrHrs") or
            q.get("lastTradePrice") or
            q.get("bidPrice") or 0
        )
        return (spot if spot > 0 else None), sym_id
    except Exception as exc:
        log.warning("Questrade spot fetch failed for %s: %s", underlying, exc)
        return None, None


def _spot_from_yfinance(underlying: str) -> Optional[float]:
    try:
        import yfinance as yf
        import warnings; warnings.filterwarnings("ignore")
        h = yf.Ticker(underlying).history(period="2d")
        return float(h["Close"].iloc[-1]) if not h.empty else None
    except Exception as exc:
        log.warning("yfinance spot fetch failed for %s: %s", underlying, exc)
        return None


def _option_mid_from_questrade(client, underlying_id: int,
                                option_type: str, strike: float,
                                expiry: str) -> tuple:
    """
    Returns (bid, ask, mid) or (None, None, None).
    Walks the option chain to find the specific contract.
    """
    try:
        chain = client.option_chain(underlying_id)
        target_date = expiry[:10]   # "YYYY-MM-DD"
        is_call = option_type.lower() == "call"

        for exp_group in chain:
            exp_raw = exp_group.get("expiryDate", "")[:10]
            if exp_raw != target_date:
                continue
            for root in exp_group.get("chainPerRoot", []):
                for strike_row in root.get("chainPerStrikePrice", []):
                    if abs(float(strike_row.get("strikePrice", 0)) - strike) < 0.01:
                        opt_id = (strike_row["callSymbolId"] if is_call
                                  else strike_row["putSymbolId"])
                        quotes = client.option_quotes(option_ids=[opt_id])
                        if quotes:
                            q = quotes[0]
                            bid = float(q.get("bidPrice") or 0)
                            ask = float(q.get("askPrice") or 0)
                            if bid > 0 and ask > 0:
                                return bid, ask, (bid + ask) / 2
                            last = float(q.get("lastTradePriceTrHrs") or
                                         q.get("lastTradePrice") or 0)
                            if last > 0:
                                return None, None, last
        return None, None, None
    except Exception as exc:
        log.warning("Questrade option chain fetch failed: %s", exc)
        return None, None, None


def _option_mid_from_yfinance(underlying: str, option_type: str,
                               strike: float, expiry: str) -> tuple:
    """Returns (bid, ask, mid) or (None, None, None)."""
    try:
        import yfinance as yf
        import warnings; warnings.filterwarnings("ignore")
        t = yf.Ticker(underlying)
        chain = t.option_chain(expiry)
        df = chain.calls if option_type.lower() == "call" else chain.puts
        row = df[abs(df["strike"] - strike) < 0.01]
        if row.empty:
            # nearest strike
            row = df.iloc[(df["strike"] - strike).abs().argsort()[:1]]
        if row.empty:
            return None, None, None
        r   = row.iloc[0]
        bid = float(r.get("bid", 0) or 0)
        ask = float(r.get("ask", 0) or 0)
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else float(r.get("lastPrice", 0) or 0)
        return (bid or None), (ask or None), (mid if mid > 0 else None)
    except Exception as exc:
        log.warning("yfinance option fetch failed: %s", exc)
        return None, None, None


# ── IV inversion + Greeks (identical to pricer.py) ────────────────────────────

def _invert_and_price(spot, strike, r, T, is_call, mid):
    """Returns (iv, res_dict) or (None, None)."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
    import quantcore

    type_int = 0 if is_call else 1
    sigma = max(0.01, min(mid / max(spot * math.sqrt(T / (2 * math.pi)), 1e-9), 5.0))
    for _ in range(150):
        res = quantcore.bs_full(type_int, float(spot), float(strike),
                                float(r), float(sigma), float(T))
        p, v = res["price"], res["vega"]
        if v < 1e-12:
            return None, None
        sigma -= (p - mid) / v
        sigma = max(0.001, min(sigma, 10.0))
        if abs(p - mid) < 1e-7:
            return sigma, res
    return sigma, res   # best estimate even if not fully converged


# ── main entry point ──────────────────────────────────────────────────────────

def analyse_hypothetical(client, spec: HypotheticalSpec) -> HypotheticalResult:
    """
    Run a hypothetical contract through the full live-data + C++ engine path.
    """
    # ── T (time to expiry) ────────────────────────────────────────────────────
    try:
        exp_dt = datetime.strptime(spec.expiry, "%Y-%m-%d")
        T = max((exp_dt - datetime.utcnow()).days / 365.0, 1e-4)
    except Exception:
        return HypotheticalResult(spec=spec, spot=None, spot_source="unavailable",
                                  market_bid=None, market_ask=None, market_mid=None,
                                  mid_source="unavailable", T_years=None,
                                  implied_vol=None, model_price=None,
                                  delta=None, gamma=None, theta_day=None, vega=None,
                                  error="Could not parse expiry date")

    # ── spot ──────────────────────────────────────────────────────────────────
    spot, underlying_id = _spot_from_questrade(client, spec.underlying)
    spot_source = "questrade"
    if spot is None:
        spot = _spot_from_yfinance(spec.underlying)
        spot_source = "yfinance" if spot else "unavailable"

    if spot is None:
        return HypotheticalResult(spec=spec, spot=None, spot_source="unavailable",
                                  market_bid=None, market_ask=None, market_mid=None,
                                  mid_source="unavailable", T_years=T,
                                  implied_vol=None, model_price=None,
                                  delta=None, gamma=None, theta_day=None, vega=None,
                                  error=f"Could not fetch spot for {spec.underlying}")

    # ── option mid ────────────────────────────────────────────────────────────
    bid, ask, mid = None, None, None
    mid_source = "unavailable"

    if underlying_id:
        bid, ask, mid = _option_mid_from_questrade(
            client, underlying_id, spec.option_type, spec.strike, spec.expiry)
        if mid:
            mid_source = "questrade"

    if mid is None:
        bid, ask, mid = _option_mid_from_yfinance(
            spec.underlying, spec.option_type, spec.strike, spec.expiry)
        if mid:
            mid_source = "yfinance"

    if mid is None:
        return HypotheticalResult(spec=spec, spot=spot, spot_source=spot_source,
                                  market_bid=bid, market_ask=ask, market_mid=None,
                                  mid_source="unavailable", T_years=T,
                                  implied_vol=None, model_price=None,
                                  delta=None, gamma=None, theta_day=None, vega=None,
                                  error=f"No market quote for {spec.underlying} "
                                        f"{spec.option_type} K={spec.strike} {spec.expiry}")

    # ── IV + Greeks ───────────────────────────────────────────────────────────
    is_call = spec.option_type.lower() == "call"
    try:
        iv, res = _invert_and_price(spot, spec.strike, RISK_FREE, T, is_call, mid)
    except Exception as exc:
        return HypotheticalResult(spec=spec, spot=spot, spot_source=spot_source,
                                  market_bid=bid, market_ask=ask, market_mid=mid,
                                  mid_source=mid_source, T_years=T,
                                  implied_vol=None, model_price=None,
                                  delta=None, gamma=None, theta_day=None, vega=None,
                                  error=f"Engine error: {exc}")

    if iv is None or res is None:
        return HypotheticalResult(spec=spec, spot=spot, spot_source=spot_source,
                                  market_bid=bid, market_ask=ask, market_mid=mid,
                                  mid_source=mid_source, T_years=T,
                                  implied_vol=None, model_price=None,
                                  delta=None, gamma=None, theta_day=None, vega=None,
                                  error="IV inversion failed (vega near zero — deep ITM/OTM?)")

    return HypotheticalResult(
        spec        = spec,
        spot        = spot,
        spot_source = spot_source,
        market_bid  = bid,
        market_ask  = ask,
        market_mid  = mid,
        mid_source  = mid_source,
        T_years     = T,
        implied_vol = iv,
        model_price = res["price"],
        delta       = res["delta"],
        gamma       = res["gamma"],
        theta_day   = res["theta"] / 365.0,
        vega        = res["vega"],
    )
