"""
Position pricer — feeds live Questrade positions into the QuantCore C++ engine.

For each option position:
  1. Resolve the option's details (underlying, strike, expiry, type) from
     the Questrade option chain.
  2. Fetch current underlying spot price.
  3. Invert Black-Scholes on the current option mid-price to recover
     market-implied vol per strike.
  4. Run bs_full() (C++ binding) to get model price + all four Greeks.
  5. Compute P&L vs position cost basis if available.

Equity (non-option) positions get delta=1 / gamma=theta=vega=0.
"""

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

log = logging.getLogger(__name__)

# ── risk-free rate proxy ──────────────────────────────────────────────────────
# Configurable; defaults to 4.5% (approximate US 3-month T-bill, 2026).
# A more precise value can be fetched from a fixed-income endpoint, but
# BS sensitivity to ±50bp in r is small compared to vol uncertainty.
DEFAULT_RISK_FREE = 0.045


# ── data classes ─────────────────────────────────────────────────────────────

@dataclass
class PositionGreeks:
    symbol:        str
    description:   str
    qty:           float          # + long, - short
    mkt_price:     float          # current mid-price per share/contract
    mkt_value:     float          # qty × mkt_price × multiplier
    model_price:   Optional[float]  # BS price using market IV
    implied_vol:   Optional[float]  # recovered IV
    delta:         float           # per-share delta (raw, not dollar)
    gamma:         float
    theta_day:     float           # per calendar day
    vega:          float           # per 1 vol-point (0.01 σ)
    is_option:     bool
    option_type:   Optional[str]   # 'call' or 'put'
    underlying:    Optional[str]
    strike:        Optional[float]
    expiry:        Optional[str]
    T_years:       Optional[float]
    error:         Optional[str] = None   # non-None if pricing failed


@dataclass
class PortfolioSnapshot:
    timestamp:       str
    account_id:      str
    account_type:    str
    positions:       List[PositionGreeks] = field(default_factory=list)
    nav:             float = 0.0
    cash:            float = 0.0
    # portfolio-level aggregates (set by aggregate())
    net_delta_usd:   float = 0.0   # Σ delta × spot × qty × multiplier
    net_gamma:       float = 0.0
    net_theta_day:   float = 0.0
    net_vega_pct:    float = 0.0   # Σ vega × qty × multiplier × 0.01
    var_hist_1d:     Optional[float] = None
    var_param_1d:    Optional[float] = None

    def aggregate(self) -> None:
        """Roll up per-position Greeks to portfolio level."""
        self.net_delta_usd  = 0.0
        self.net_gamma      = 0.0
        self.net_theta_day  = 0.0
        self.net_vega_pct   = 0.0
        for p in self.positions:
            mult = 100 if p.is_option else 1
            self.net_delta_usd += p.delta   * (p.mkt_price or 0) * p.qty * mult
            self.net_gamma     += p.gamma   * p.qty * mult
            self.net_theta_day += p.theta_day * p.qty * mult
            self.net_vega_pct  += p.vega    * p.qty * mult * 0.01


# ── IV inversion (reuses the C++ bs_full from pybind11) ──────────────────────

def _invert_iv(mid: float, S: float, K: float, r: float, T: float,
               is_call: bool, tol: float = 1e-6) -> Optional[float]:
    """Newton-Raphson IV recovery using C++ bs_full."""
    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
        import quantcore
    except ImportError:
        return None

    if T <= 0 or mid < 1e-4:
        return None

    type_int = 0 if is_call else 1
    sigma = max(0.01, min(mid / max(S * math.sqrt(T / (2 * math.pi)), 1e-9), 5.0))

    for _ in range(150):
        try:
            res = quantcore.bs_full(type_int, float(S), float(K), float(r),
                                    float(sigma), float(T))
        except Exception:
            return None
        p, v = res['price'], res['vega']
        if v < 1e-12:
            return None
        sigma -= (p - mid) / v
        sigma = max(0.001, min(sigma, 10.0))
        if abs(p - mid) < tol:
            return sigma
    return sigma


# ── main pricer ───────────────────────────────────────────────────────────────

class PositionPricer:
    """
    Prices a list of Questrade positions through the QuantCore C++ engine.

    Parameters
    ----------
    client : QuestradeClient
        An authenticated, read-only Questrade client.
    risk_free : float
        Annualised risk-free rate proxy.
    """

    def __init__(self, client, risk_free: float = DEFAULT_RISK_FREE):
        self._client    = client
        self._risk_free = risk_free
        self._spot_cache: dict = {}   # symbol_id → spot price

    def price_account(self, account_id: str, account_type: str) -> PortfolioSnapshot:
        positions_raw = self._client.positions(account_id)
        balances_raw  = self._client.balances(account_id)

        cash = sum(b.get("cash", 0) for b in balances_raw.get("combinedBalances", []))
        nav  = sum(b.get("totalEquity", 0) for b in balances_raw.get("combinedBalances", []))

        snap = PortfolioSnapshot(
            timestamp    = datetime.now().isoformat(timespec="seconds"),
            account_id   = account_id,
            account_type = account_type,
            nav          = nav,
            cash         = cash,
        )

        for raw in positions_raw:
            pg = self._price_one(raw)
            snap.positions.append(pg)

        snap.aggregate()
        return snap

    def _price_one(self, raw: dict) -> PositionGreeks:
        sym_id    = raw.get("symbolId", 0)
        symbol    = raw.get("symbol", "?")
        desc      = raw.get("description", "")
        qty       = float(raw.get("openQuantity", 0))
        mkt_price = float(raw.get("currentPrice", 0) or 0)
        mkt_value = float(raw.get("currentMarketValue", 0) or 0)

        is_option = raw.get("securityType", "") in ("Option", "MutualFund") or \
                    any(c in symbol for c in (" Call", " Put", "C ", "P "))

        if not is_option:
            return PositionGreeks(
                symbol=symbol, description=desc,
                qty=qty, mkt_price=mkt_price, mkt_value=mkt_value,
                model_price=mkt_price, implied_vol=None,
                delta=1.0, gamma=0.0, theta_day=0.0, vega=0.0,
                is_option=False, option_type=None, underlying=None,
                strike=None, expiry=None, T_years=None,
            )

        return self._price_option(raw, symbol, desc, qty, mkt_price, mkt_value)

    def _price_option(self, raw: dict, symbol: str, desc: str,
                      qty: float, mkt_price: float, mkt_value: float
                      ) -> PositionGreeks:
        """Resolve option details → get spot → invert IV → compute Greeks."""
        try:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
            import quantcore
        except ImportError as e:
            return self._error_position(symbol, desc, qty, mkt_price, mkt_value,
                                        f"quantcore import failed: {e}")
        try:
            sym_id  = raw.get("symbolId", 0)
            sym_info = self._client.symbol(sym_id)

            underlying_id   = sym_info.get("underlyingId")
            strike          = float(sym_info.get("strikePrice", 0) or 0)
            expiry_str      = sym_info.get("expiryDate", "")[:10]  # YYYY-MM-DD
            option_type_raw = sym_info.get("optionType", "").lower()
            is_call         = option_type_raw == "call"
            underlying_sym  = sym_info.get("underlying", symbol)

            # Time to expiry in years
            try:
                expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d")
                T = max((expiry_dt - datetime.utcnow()).days / 365.0, 1e-4)
            except Exception:
                T = 0.1  # fallback

            # Underlying spot
            spot = self._get_spot(underlying_id)

            # Mid-price for IV inversion
            mid = mkt_price

            # Invert IV
            iv = _invert_iv(mid, spot, strike, self._risk_free, T, is_call)
            if iv is None or iv < 0.01:
                raise ValueError(f"IV inversion failed (mid={mid:.3f}, S={spot:.2f})")

            # Full BS
            type_int = 0 if is_call else 1
            res = quantcore.bs_full(type_int, spot, strike, self._risk_free, iv, T)

            return PositionGreeks(
                symbol=symbol, description=desc,
                qty=qty, mkt_price=mkt_price, mkt_value=mkt_value,
                model_price=res["price"], implied_vol=iv,
                delta=res["delta"], gamma=res["gamma"],
                theta_day=res["theta"] / 365.0,
                vega=res["vega"],
                is_option=True,
                option_type="call" if is_call else "put",
                underlying=underlying_sym,
                strike=strike, expiry=expiry_str, T_years=T,
            )

        except Exception as exc:
            log.warning("Option pricing failed for %s: %s", symbol, exc)
            return self._error_position(symbol, desc, qty, mkt_price, mkt_value,
                                        str(exc))

    def _get_spot(self, symbol_id: int) -> float:
        if symbol_id in self._spot_cache:
            return self._spot_cache[symbol_id]
        q = self._client.quote(symbol_id)
        spot = float(q.get("lastTradePriceTrHrs") or
                     q.get("lastTradePrice") or
                     q.get("bidPrice") or 0)
        self._spot_cache[symbol_id] = spot
        return spot

    @staticmethod
    def _error_position(symbol, desc, qty, mkt_price, mkt_value, err) -> PositionGreeks:
        return PositionGreeks(
            symbol=symbol, description=desc,
            qty=qty, mkt_price=mkt_price, mkt_value=mkt_value,
            model_price=None, implied_vol=None,
            delta=0.0, gamma=0.0, theta_day=0.0, vega=0.0,
            is_option=True, option_type=None, underlying=None,
            strike=None, expiry=None, T_years=None,
            error=err,
        )
