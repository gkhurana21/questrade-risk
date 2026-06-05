#!/usr/bin/env python3
"""
QuantCore hypothetical position analyser — READ-ONLY, analysis only.

Specify one or more contracts manually.  For each contract the tool:
  • Fetches live spot from Questrade (yfinance fallback)
  • Fetches real bid/ask from Questrade option chain (yfinance fallback)
  • Inverts market-implied vol via the C++ Black-Scholes engine
  • Computes delta / gamma / theta / vega via the same C++ engine
  • Runs portfolio VaR on the combined underlying exposures

Everything is labelled HYPOTHETICAL.  No orders are placed or staged.

Usage
-----
    python3 run_hypothetical.py \\
        "AAPL,call,230,2026-07-18,1" \\
        "SPY,put,560,2026-06-20,-2"

Contract format:  UNDERLYING,call|put,STRIKE,YYYY-MM-DD,QTY
  QTY: positive = long, negative = short
"""

import sys, os, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("yfinance").setLevel(logging.WARNING)

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from questrade import TokenManager, QuestradeClient
from analysis.hypothetical import HypotheticalSpec, analyse_hypothetical
from analysis.var_live import compute_var
from analysis.pricer import PortfolioSnapshot, PositionGreeks


def parse_spec(s: str) -> HypotheticalSpec:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 5:
        raise ValueError(f"Expected UNDERLYING,call|put,STRIKE,YYYY-MM-DD,QTY — got: {s!r}")
    underlying, opt_type, strike, expiry, qty = parts
    if opt_type.lower() not in ("call", "put"):
        raise ValueError(f"option type must be 'call' or 'put', got {opt_type!r}")
    return HypotheticalSpec(
        underlying  = underlying.upper(),
        option_type = opt_type.lower(),
        strike      = float(strike),
        expiry      = expiry,
        qty         = int(qty),
    )


def render_result(r) -> str:
    lines = []
    a = lines.append
    bar = "─" * 68

    a("")
    a(f"  ╔{'═'*66}╗")
    a(f"  ║  HYPOTHETICAL POSITION — what-if analysis on real market data  ║")
    a(f"  ╚{'═'*66}╝")
    a(f"  {r.spec.label()}")
    a(f"  {bar}")

    if r.error:
        a(f"  ✗ Could not price: {r.error}")
        a(f"  {bar}")
        return "\n".join(lines)

    dte = int(r.T_years * 365) if r.T_years else "?"

    # ── inputs ────────────────────────────────────────────────────────────────
    a(f"  INPUTS (live market data)")
    a(f"    Underlying spot   : ${r.spot:>9.2f}  [{r.spot_source}]")
    a(f"    Strike            : ${r.spec.strike:>9.2f}")
    a(f"    Days to expiry    :  {dte} days  (T = {r.T_years:.4f} yr)")
    a(f"    Risk-free rate    :  4.50%  (fixed proxy)")
    if r.market_bid and r.market_ask:
        a(f"    Market bid / ask  : ${r.market_bid:.3f} / ${r.market_ask:.3f}  [{r.mid_source}]")
    a(f"    Mid-price used    : ${r.market_mid:.3f}  [{r.mid_source}]")

    # ── C++ engine output ─────────────────────────────────────────────────────
    a("")
    a(f"  QuantCore C++ engine output")
    a(f"    Implied vol (IV)  :  {r.implied_vol*100:>6.2f}%")
    model_vs_mid = r.model_price - r.market_mid if r.model_price and r.market_mid else 0
    a(f"    Model price       : ${r.model_price:>8.4f}  "
      f"(vs market mid ${r.market_mid:.3f}  Δ={model_vs_mid:+.4f})")
    is_c   = r.spec.option_type.lower() == "call"
    d_ok   = (0 <= r.delta <= 1) if is_c else (-1 <= r.delta <= 0)
    d_range = "call: [0,1]" if is_c else "put: [-1,0]"
    a(f"    Delta (Δ)         : {r.delta:>+9.4f}  "
      f"{'✓' if d_ok else '⚠ OOR'}  ({d_range})")
    a(f"    Gamma (Γ)         : {r.gamma:>10.6f}")
    a(f"    Theta/day (Θ)     : {r.theta_day:>+10.4f}  $/share/day")
    a(f"    Vega (ν)          : {r.vega:>10.4f}  $/share per unit σ")

    # ── position-level P&L sensitivity ────────────────────────────────────────
    mult = 100
    a("")
    a(f"  Position-level sensitivities  (qty={r.spec.qty:+d}, {abs(r.spec.qty)} contract(s))")
    dollar_delta = r.delta * r.spec.qty * mult * r.spot
    dollar_theta = r.theta_day * r.spec.qty * mult
    dollar_vega  = r.vega * r.spec.qty * mult * 0.01
    a(f"    Dollar delta      : ${dollar_delta:>+10,.0f}  (P&L per $1 move in {r.spec.underlying})")
    a(f"    Daily theta cost  : ${dollar_theta:>+10,.2f}  (time decay per calendar day)")
    a(f"    Vega per 1% σ     : ${dollar_vega:>+10,.2f}  (P&L per 1% change in IV)")

    # ── sanity check ──────────────────────────────────────────────────────────
    flags = r.sanity_flags()
    a("")
    if flags:
        a(f"  ⚠  SANITY WARNINGS — investigate before trusting numbers:")
        for f in flags:
            a(f"      • {f}")
    else:
        a(f"  ✓  Sanity check passed  "
          f"({'call' if r.spec.option_type=='call' else 'put'} delta in valid range, "
          f"IV {r.implied_vol*100:.1f}% reasonable, gamma ≥ 0)")

    a(f"  {bar}")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    specs = []
    for arg in sys.argv[1:]:
        try:
            specs.append(parse_spec(arg))
        except ValueError as e:
            print(f"ERROR parsing {arg!r}: {e}")
            sys.exit(1)

    tm     = TokenManager()
    client = QuestradeClient(tm)

    print(f"\n{'═'*70}")
    print(f"  QuantCore Hypothetical Analysis  [READ-ONLY, ANALYSIS ONLY]")
    print(f"  {len(specs)} contract(s) specified — fetching live market data ...")
    print(f"{'═'*70}")

    results = []
    for spec in specs:
        print(f"\n  Analysing {spec.underlying} {spec.option_type} K={spec.strike} {spec.expiry} ...")
        r = analyse_hypothetical(client, spec)
        results.append(r)
        print(render_result(r))

    # ── portfolio VaR across the hypothetical positions ────────────────────────
    if any(r.delta is not None for r in results):
        print(f"\n{'─'*70}")
        print(f"  PORTFOLIO VAR  (hypothetical positions, real historical returns)")
        print(f"{'─'*70}")

        # Build a PortfolioSnapshot to reuse the VaR module
        snap = PortfolioSnapshot(
            timestamp    = "hypothetical",
            account_id   = "HYPOTHETICAL",
            account_type = "What-if",
        )
        for r in results:
            if r.delta is None:
                continue
            snap.positions.append(PositionGreeks(
                symbol      = r.spec.label(),
                description = "",
                qty         = float(r.spec.qty),
                mkt_price   = r.market_mid or 0,
                mkt_value   = (r.market_mid or 0) * r.spec.qty * 100,
                model_price = r.model_price,
                implied_vol = r.implied_vol,
                delta       = r.delta,
                gamma       = r.gamma,
                theta_day   = r.theta_day or 0,
                vega        = r.vega,
                is_option   = True,
                option_type = r.spec.option_type,
                underlying  = r.spec.underlying,
                strike      = r.spec.strike,
                expiry      = r.spec.expiry,
                T_years     = r.T_years,
            ))

        snap.aggregate()
        compute_var(snap)

        print(f"  Net delta exposure : ${snap.net_delta_usd:>+12,.0f}")
        print(f"  Net theta/day      : ${snap.net_theta_day:>+12,.2f}")
        print(f"  Net vega per 1% σ  : ${snap.net_vega_pct:>+12,.2f}")
        if snap.var_hist_1d is not None:
            print(f"  Historical 95% VaR : ${snap.var_hist_1d:>10,.0f}  (1-day, real returns)")
            print(f"  Parametric 95% VaR : ${snap.var_param_1d:>10,.0f}")
        else:
            print("  VaR: insufficient historical return data.")

    print(f"\n{'═'*70}")
    print(f"  HYPOTHETICAL ANALYSIS COMPLETE — no orders placed or staged.")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
