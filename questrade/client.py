"""
Read-only Questrade REST API client.

DELIBERATELY ABSENT: This module does not implement, import, or reference any
Questrade order-placement endpoints.  Specifically absent:
  - POST /v1/accounts/{id}/orders
  - PUT  /v1/accounts/{id}/orders/{id}
  - DELETE /v1/accounts/{id}/orders/{id}
  - Any "executions" write path

The POST used in option_quotes() is a READ operation: it queries the current
bid/ask for a list of option contracts.  It does not place, modify, or cancel
any order — Questrade requires POST because the filter body is complex.

Every method here is a query.  Nothing changes state on the brokerage side.
"""

import logging
from typing import List, Optional

import requests
from .token_manager import TokenManager

log = logging.getLogger(__name__)


class QuestradeClient:
    """
    Read-only Questrade REST client.
    Retries once on HTTP 401 (access token expiry) before raising.
    """

    def __init__(self, token_manager: TokenManager):
        self._tm = token_manager

    # ── internal ──────────────────────────────────────────────────────────────

    def _get(self, path: str, **params) -> dict:
        return self._request("GET", path, params=params or None)

    def _post_query(self, path: str, body: dict) -> dict:
        """
        POST used solely as a query carrier (no state change).
        Option quotes require a POST because the filter body is too large for a
        query string — Questrade's design, not an order operation.
        """
        return self._request("POST", path, json=body)

    def _request(self, method: str, path: str, **kwargs) -> dict:
        for attempt in range(2):
            token, server = self._tm.get_access_token()
            url = f"{server}v1/{path}"
            log.debug("Questrade %s %s", method, url)
            try:
                resp = requests.request(
                    method, url,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=15,
                    **kwargs,
                )
            except requests.RequestException as exc:
                raise RuntimeError(
                    f"Questrade API network error on {method} {path}: "
                    f"{type(exc).__name__}"
                ) from exc

            if resp.status_code == 401 and attempt == 0:
                log.info("Questrade 401 — forcing token refresh")
                self._tm.invalidate()
                continue

            if not resp.ok:
                raise RuntimeError(
                    f"Questrade API error: HTTP {resp.status_code} on "
                    f"{method} {path} — {resp.text[:200]}"
                )

            return resp.json()

        raise RuntimeError("Questrade authentication failed after token refresh")

    # ── account read-only ─────────────────────────────────────────────────────

    def accounts(self) -> List[dict]:
        """List all accounts linked to this token."""
        return self._get("accounts")["accounts"]

    def positions(self, account_id: str) -> List[dict]:
        """Current open positions for an account."""
        return self._get(f"accounts/{account_id}/positions")["positions"]

    def balances(self, account_id: str) -> dict:
        """Combined and per-currency balances."""
        return self._get(f"accounts/{account_id}/balances")

    def activities(self, account_id: str,
                   start_time: str, end_time: str) -> List[dict]:
        """Account activity (trades, dividends, etc.) — read-only history."""
        return self._get(
            f"accounts/{account_id}/activities",
            startTime=start_time, endTime=end_time,
        )["activities"]

    # ── market data read-only ─────────────────────────────────────────────────

    def quotes(self, symbol_ids: List[int]) -> List[dict]:
        """Current bid/ask/last for a list of equities or futures."""
        ids = ",".join(str(i) for i in symbol_ids)
        return self._get("markets/quotes", ids=ids)["quotes"]

    def quote(self, symbol_id: int) -> dict:
        """Convenience single-symbol wrapper."""
        return self.quotes([symbol_id])[0]

    def symbol(self, symbol_id: int) -> dict:
        """Symbol metadata: name, description, exchange, secType."""
        return self._get(f"symbols/{symbol_id}")["symbols"][0]

    def symbol_search(self, prefix: str, offset: int = 0) -> List[dict]:
        """Find symbols by prefix."""
        return self._get("symbols/search", prefix=prefix, offset=offset)["symbols"]

    def option_chain(self, underlying_id: int) -> List[dict]:
        """
        Option chain for an underlying equity.
        Returns expiry groups with strike/call/put symbolId mapping.
        """
        return self._get(f"symbols/{underlying_id}/options")["optionChain"]

    def option_quotes(self, filters: Optional[List[dict]] = None,
                      option_ids: Optional[List[int]] = None) -> List[dict]:
        """
        Current quotes for specific option contracts.

        Supply either:
          - filters: list of {underlyingId, expiryDate, optionType, minStrikePrice,
                               maxStrikePrice}
          - option_ids: list of option symbolIds

        This uses HTTP POST as a query body carrier (Questrade API requirement);
        it does not place any order.
        """
        body: dict = {}
        if filters:
            body["filters"] = filters
        if option_ids:
            body["optionIds"] = option_ids
        if not body:
            raise ValueError("Provide filters or option_ids")
        return self._post_query("markets/quotes/options", body)["optionQuotes"]

    def market_hours(self, market_id: int) -> dict:
        """Trading hours for a market."""
        return self._get(f"markets/{market_id}")

    def candles(self, symbol_id: int, start_time: str, end_time: str,
                interval: str = "OneDay") -> List[dict]:
        """Historical OHLCV candles — read-only price history."""
        return self._get(
            f"markets/candles/{symbol_id}",
            startTime=start_time, endTime=end_time, interval=interval,
        )["candles"]
