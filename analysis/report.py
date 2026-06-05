"""
Read-only analysis report.

Prints a structured snapshot of portfolio risk to stdout.
Every output section is labelled as an OBSERVATION.
This module has no mechanism to produce trade instructions —
there is no order-placement code, no "recommended action" field,
and no output that says "buy" or "sell."
"""

from .pricer import PortfolioSnapshot, PositionGreeks


def _bar(ch="─", width=70):
    return ch * width


def _fmt_optional(v, fmt):
    return fmt.format(v) if v is not None else " n/a"


def render(snap: PortfolioSnapshot) -> str:
    lines = []
    a = lines.append

    a("")
    a(_bar("═"))
    a(f"  QuantCore Live Analysis — {snap.timestamp}  [READ-ONLY]")
    a(f"  Account: {snap.account_id}  ({snap.account_type})")
    a(_bar("═"))

    # ── balances ─────────────────────────────────────────────────────────────
    a("")
    a(f"  NAV: ${snap.nav:>12,.2f}    Cash: ${snap.cash:>12,.2f}")

    # ── per-position table ───────────────────────────────────────────────────
    a("")
    a(_bar())
    a("  POSITIONS")
    a(_bar())

    if not snap.positions:
        a("  (no positions)")
    else:
        hdr = (f"  {'Symbol':<28} {'Qty':>6}  {'Mkt $':>8}  "
               f"{'IV%':>5}  {'Δ':>7}  {'Γ':>8}  "
               f"{'Θ/day':>7}  {'ν':>8}")
        a(hdr)
        a("  " + "─" * 68)
        for p in snap.positions:
            if p.error:
                a(f"  {p.symbol:<28} {'':>6}  {'':>8}  {'':>5}  "
                  f"  [pricing error: {p.error[:35]}]")
                continue
            iv_str = f"{p.implied_vol*100:.1f}" if p.implied_vol else " ---"
            a(f"  {p.symbol:<28} {p.qty:>+6.0f}  {p.mkt_price:>8.3f}  "
              f"{iv_str:>5}  {p.delta:>+7.4f}  {p.gamma:>8.5f}  "
              f"{p.theta_day:>+7.4f}  {p.vega:>8.4f}")

    # ── portfolio greeks ─────────────────────────────────────────────────────
    a("")
    a(_bar())
    a("  PORTFOLIO GREEKS  (observation only — no action implied)")
    a(_bar())
    a(f"  Net delta exposure (Σ Δ·S·qty·mult) : ${snap.net_delta_usd:>+12,.0f}")
    a(f"  Net theta decay (Σ Θ·qty·mult/day)  : ${snap.net_theta_day:>+12,.2f} / day")
    a(f"  Net vega (Σ ν·qty·mult × 1% vol)    : ${snap.net_vega_pct:>+12,.2f} / 1% σ")

    # ── VaR ──────────────────────────────────────────────────────────────────
    a("")
    a(_bar())
    a("  PORTFOLIO VAR  95% confidence, 1-day  (observation only)")
    a(_bar())
    if snap.var_hist_1d is not None:
        a(f"  Historical simulation               : ${snap.var_hist_1d:>10,.0f}")
        a(f"  Parametric (normal)                 : ${snap.var_param_1d:>10,.0f}")
        a(f"  As % of NAV (historical)            : "
          f"{snap.var_hist_1d / max(snap.nav, 1) * 100:>+8.2f}%")
    else:
        a("  Insufficient data for VaR (need ≥252 days of return history).")

    # ── observations ─────────────────────────────────────────────────────────
    a("")
    a(_bar())
    a("  OBSERVATIONS  (informational — no buy/sell instructions)")
    a(_bar())
    _observations(snap, lines)

    a("")
    a(_bar("═"))
    a("  End of read-only analysis report.")
    a(_bar("═"))

    return "\n".join(lines)


def _observations(snap: PortfolioSnapshot, lines: list) -> None:
    """Generate factual, direction-free observations about the portfolio."""
    a = lines.append

    if snap.net_delta_usd != 0:
        direction = "long" if snap.net_delta_usd > 0 else "short"
        a(f"  • Net delta is {direction} ${abs(snap.net_delta_usd):,.0f} equity equivalent.")

    if snap.net_theta_day < 0:
        a(f"  • Theta decay costs ${abs(snap.net_theta_day):,.2f}/day at current positions.")
    elif snap.net_theta_day > 0:
        a(f"  • Portfolio earns ${snap.net_theta_day:,.2f}/day in theta (net short premium).")

    if snap.net_vega_pct != 0:
        direction = "gains" if snap.net_vega_pct > 0 else "loses"
        a(f"  • Portfolio {direction} ${abs(snap.net_vega_pct):,.0f} per 1% rise in implied vol.")

    if snap.var_hist_1d is not None and snap.nav > 0:
        pct = snap.var_hist_1d / snap.nav * 100
        a(f"  • Historical 95% VaR is {pct:.2f}% of NAV (${snap.var_hist_1d:,.0f}).")
        if pct > 5:
            a("    Note: VaR exceeds 5% of NAV — concentration or leverage is elevated.")

    errors = [p for p in snap.positions if p.error]
    if errors:
        a(f"  • {len(errors)} position(s) could not be priced: "
          f"{', '.join(p.symbol for p in errors[:3])}.")

    if not [l for l in lines if l.startswith("  •")]:
        a("  • No notable observations at this time.")
