# questrade-risk

Read-only Questrade API integration for the QuantCore pricing and risk engine.

**Scope:** analysis only. This tool has no order-placement capability. Every
output is labelled `[READ-ONLY]`. No buy/sell instructions are generated.

## What it does

- Connects to the Questrade API (read-only OAuth token) to fetch live positions,
  balances, and option chain quotes
- Feeds real market data into the QuantCore C++ Black-Scholes engine to compute
  per-position Greeks (delta, gamma, theta, vega) and implied volatility
- Runs portfolio-level VaR (historical simulation + parametric) against real
  historical returns via yfinance
- Supports a **hypothetical position mode**: specify any contract manually and
  price it against live market data without holding it

## Setup

1. Install dependencies:
   ```bash
   pip install requests python-dotenv yfinance numpy pandas
   ```

2. The QuantCore C++ module (`quantcore.cpython-*.so`) must be built and on the
   Python path. See [quantcore](https://github.com/gkhurana21/quantcore) for
   build instructions.

3. Generate a **read-only** Questrade refresh token:
   - Log into questrade.com → My Account → Security → Connected Apps & Devices
   - Register a personal app — select **Read** scope only, not Trade
   - Copy the token (shown once)

4. Add it to `.env` (never commit this file):
   ```bash
   cp .env.template .env
   # edit .env and set QUESTRADE_REFRESH_TOKEN=<your token>
   ```

## Usage

```bash
# Live account analysis
python3 run_analysis.py

# Hypothetical position (real market data, no real holding)
python3 run_hypothetical.py "AAPL,call,315,2026-07-17,1" "SPY,put,750,2026-06-26,-2"
```

## Token rotation

Questrade refresh tokens are single-use and rotate on every authentication.
`token_manager.py` writes the new token back to `.env` before making any API
call. If the script is interrupted mid-refresh, re-generate a token from the
Questrade portal.

## Security

- `.env` is gitignored and must never be committed
- Token values are never logged or printed
- No order endpoints exist anywhere in this codebase
