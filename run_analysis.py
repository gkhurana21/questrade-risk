#!/usr/bin/env python3
"""
QuantCore Live Analysis — read-only entry point.

Usage:
    python3 run_analysis.py             # live mode (requires .env token)
    python3 run_analysis.py --dry-run   # dry-run against sample data

This script is READ-ONLY.  It has no mechanism to place, modify, or cancel
any brokerage order.  All output is clearly labelled as observations.
"""

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s  %(message)s",
)

# Silence noisy third-party loggers
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("yfinance").setLevel(logging.WARNING)


# ── dry-run sample data ───────────────────────────────────────────────────────
# These are fictional positions used only when --dry-run is passed.
# They look realistic (SPY options, AAPL calls) but are not real account data.

SAMPLE_POSITIONS = [
    {
        "symbol":              "AAPL 24Jan27 220.00 C",
        "description":         "Apple Inc Call 220 Jan 2027",
        "symbolId":            123001,
        "openQuantity":        10,
        "currentPrice":        18.45,
        "currentMarketValue":  18450.0,
        "securityType":        "Option",
        # option metadata fields (normally resolved via symbol lookup)
        "_mock_underlying":    "AAPL",
        "_mock_spot":          215.80,
        "_mock_strike":        220.0,
        "_mock_expiry":        "2027-01-15",
        "_mock_type":          "call",
    },
    {
        "symbol":              "SPY 20Jun26 560.00 P",
        "description":         "SPDR S&P 500 Put 560 Jun 2026",
        "symbolId":            123002,
        "openQuantity":        -5,
        "currentPrice":        8.20,
        "currentMarketValue":  -4100.0,
        "securityType":        "Option",
        "_mock_underlying":    "SPY",
        "_mock_spot":          575.30,
        "_mock_strike":        560.0,
        "_mock_expiry":        "2026-06-20",
        "_mock_type":          "put",
    },
    {
        "symbol":              "MSFT",
        "description":         "Microsoft Corp",
        "symbolId":            123003,
        "openQuantity":        50,
        "currentPrice":        420.10,
        "currentMarketValue":  21005.0,
        "securityType":        "Equity",
    },
]

SAMPLE_BALANCES = {
    "combinedBalances": [
        {"cash": 12_340.50, "totalEquity": 52_695.50}
    ]
}


def run_dry_run() -> None:
    """Exercise the full analysis pipeline against fictional sample data."""
    import math
    from datetime import datetime
    from analysis.pricer import PortfolioSnapshot, PositionGreeks
    from analysis.var_live import compute_var
    from analysis import report

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))
    try:
        import quantcore
    except ImportError:
        print("ERROR: quantcore module not found. Build it first (CMake + make).")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  DRY RUN — sample data only, NOT connected to any live account")
    print("=" * 70)

    snap = PortfolioSnapshot(
        timestamp    = datetime.now().isoformat(timespec="seconds"),
        account_id   = "DRYRUN-XXXX",
        account_type = "Dry Run (Sample)",
        nav          = SAMPLE_BALANCES["combinedBalances"][0]["totalEquity"],
        cash         = SAMPLE_BALANCES["combinedBalances"][0]["cash"],
    )

    for raw in SAMPLE_POSITIONS:
        sec_type = raw.get("securityType", "")
        qty      = float(raw["openQuantity"])
        mkt_p    = float(raw["currentPrice"])
        mkt_v    = float(raw["currentMarketValue"])

        if sec_type == "Equity":
            snap.positions.append(PositionGreeks(
                symbol=raw["symbol"], description=raw["description"],
                qty=qty, mkt_price=mkt_p, mkt_value=mkt_v,
                model_price=mkt_p, implied_vol=None,
                delta=1.0, gamma=0.0, theta_day=0.0, vega=0.0,
                is_option=False, option_type=None, underlying=None,
                strike=None, expiry=None, T_years=None,
            ))
            continue

        # Option: use the mock fields to call into the C++ engine
        S      = float(raw["_mock_spot"])
        K      = float(raw["_mock_strike"])
        expiry = raw["_mock_expiry"]
        r      = 0.045
        is_call = raw["_mock_type"] == "call"

        try:
            exp_dt = datetime.strptime(expiry, "%Y-%m-%d")
            T      = max((exp_dt - datetime.utcnow()).days / 365.0, 1e-4)
        except Exception:
            T = 0.5

        mid     = mkt_p
        sigma   = max(0.01, min(mid / max(S * math.sqrt(T / (2 * math.pi)), 1e-9), 5.0))
        type_int = 0 if is_call else 1

        # Newton-Raphson IV
        iv = None
        for _ in range(150):
            res = quantcore.bs_full(type_int, S, K, r, sigma, T)
            p, v = res["price"], res["vega"]
            if v < 1e-12:
                break
            sigma -= (p - mid) / v
            sigma = max(0.001, min(sigma, 10.0))
            if abs(p - mid) < 1e-6:
                iv = sigma
                break

        if iv is None:
            snap.positions.append(PositionGreeks(
                symbol=raw["symbol"], description=raw["description"],
                qty=qty, mkt_price=mkt_p, mkt_value=mkt_v,
                model_price=None, implied_vol=None,
                delta=0.0, gamma=0.0, theta_day=0.0, vega=0.0,
                is_option=True, option_type=raw["_mock_type"],
                underlying=raw["_mock_underlying"],
                strike=K, expiry=expiry, T_years=T,
                error="IV inversion failed",
            ))
            continue

        res = quantcore.bs_full(type_int, S, K, r, iv, T)
        snap.positions.append(PositionGreeks(
            symbol=raw["symbol"], description=raw["description"],
            qty=qty, mkt_price=mkt_p, mkt_value=mkt_v,
            model_price=res["price"], implied_vol=iv,
            delta=res["delta"], gamma=res["gamma"],
            theta_day=res["theta"] / 365.0,
            vega=res["vega"],
            is_option=True, option_type=raw["_mock_type"],
            underlying=raw["_mock_underlying"],
            strike=K, expiry=expiry, T_years=T,
        ))

    snap.aggregate()

    # VaR: fetch real historical returns for the fictitious underlyings
    print("  Fetching real historical returns for VaR (yfinance) ...")
    compute_var(snap)

    print(report.render(snap))


def run_live() -> None:
    """Connect to the real Questrade API and price actual positions."""
    from questrade import TokenManager, QuestradeClient
    from analysis.pricer import PositionPricer
    from analysis.var_live import compute_var
    from analysis import report

    tm     = TokenManager()
    client = QuestradeClient(tm)

    print("\nFetching accounts ...")
    accounts = client.accounts()
    if not accounts:
        print("No accounts found.")
        return

    for acct in accounts:
        acct_id   = acct["number"]
        acct_type = acct.get("type", "?")
        print(f"\nPricing account {acct_id} ({acct_type}) ...")

        pricer = PositionPricer(client)
        snap   = pricer.price_account(acct_id, acct_type)

        print("Fetching historical returns for VaR ...")
        compute_var(snap)

        print(report.render(snap))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QuantCore read-only live analysis")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run against sample data (no live account needed)")
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run()
    else:
        # Check for token before attempting live connection
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
        if not os.getenv("QUESTRADE_REFRESH_TOKEN"):
            print(
                "\nERROR: QUESTRADE_REFRESH_TOKEN not found in .env\n"
                "\nTo connect to your live account:\n"
                "  1. Follow the setup instructions in .env.template\n"
                "  2. Add your read-only refresh token to .env\n"
                "  3. Run: python3 run_analysis.py\n"
                "\nTo see a dry run with sample data:\n"
                "  python3 run_analysis.py --dry-run\n"
            )
            sys.exit(1)
        run_live()


if __name__ == "__main__":
    main()
