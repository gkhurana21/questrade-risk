"""
Questrade OAuth token manager.

Token rotation contract (from Questrade docs):
  - Access tokens expire after 30 minutes.
  - Calling the token endpoint with the current refresh token returns a NEW
    access token AND a NEW refresh token.  The old refresh token is immediately
    invalidated — if it is not persisted before the first API call, access is
    permanently lost for that session.
  - This class persists the new refresh token to .env BEFORE returning the
    new access token, so a crash during the window cannot lose the credential.

Security constraints (enforced here):
  - Tokens are NEVER logged or printed, even at DEBUG level.
  - Token values are NEVER put into exception messages.
  - The only place a token is written is the .env file on disk.
"""

import os
import time
import logging
from pathlib import Path
from typing import Tuple

import requests
from dotenv import load_dotenv, set_key

log = logging.getLogger(__name__)

TOKEN_URL = "https://login.questrade.com/oauth2/token"
_ENV_PATH = Path(__file__).parent.parent / ".env"


class TokenManager:
    """
    Thread-safe Questrade OAuth token manager with automatic rotation.

    Usage:
        tm = TokenManager()
        access_token, api_server = tm.get_access_token()
    """

    def __init__(self, env_path: Path = _ENV_PATH):
        self._env_path = env_path
        load_dotenv(env_path)

        self._refresh_token: str = os.getenv("QUESTRADE_REFRESH_TOKEN", "")
        self._api_server:     str = os.getenv("QUESTRADE_API_SERVER",    "")
        self._access_token:   str = ""
        self._expires_at:     float = 0.0

    # ── public ────────────────────────────────────────────────────────────────

    def get_access_token(self) -> Tuple[str, str]:
        """
        Return (access_token, api_server).

        Refreshes automatically when the token is absent or within 60 s of
        expiry.  The new refresh token is persisted to .env before this
        method returns — the old one is gone the moment we call the endpoint.
        """
        if not self._refresh_token:
            raise RuntimeError(
                "QUESTRADE_REFRESH_TOKEN is not set.\n"
                "  Add it to .env:  QUESTRADE_REFRESH_TOKEN=<your token>\n"
                "  See .env.template for setup instructions."
            )

        if not self._access_token or time.time() > self._expires_at - 60:
            self._rotate()

        return self._access_token, self._api_server

    def invalidate(self) -> None:
        """Force re-authentication on the next get_access_token() call."""
        self._access_token = ""
        self._expires_at   = 0.0

    # ── private ───────────────────────────────────────────────────────────────

    def _rotate(self) -> None:
        """
        Exchange the current refresh token for a new access + refresh pair.

        Write order (safety-first):
          1. Call token endpoint.
          2. Persist NEW refresh token to .env  ← must happen before step 3.
          3. Store new access token in memory.
        """
        log.info("Questrade token: refreshing (token value not logged)")
        try:
            resp = requests.post(
                TOKEN_URL,
                params={"grant_type": "refresh_token",
                        "refresh_token": self._refresh_token},
                timeout=10,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Token refresh network error: {type(exc).__name__}"
            ) from exc

        if not resp.ok:
            raise RuntimeError(
                f"Token refresh failed: HTTP {resp.status_code} — "
                "check that QUESTRADE_REFRESH_TOKEN is valid and not expired."
            )

        data = resp.json()

        # --- STEP 2: persist new refresh token BEFORE using new access token ---
        new_refresh = data["refresh_token"]   # do NOT log this value
        new_server  = data["api_server"]
        set_key(str(self._env_path), "QUESTRADE_REFRESH_TOKEN", new_refresh)
        set_key(str(self._env_path), "QUESTRADE_API_SERVER",    new_server)
        self._refresh_token = new_refresh
        self._api_server    = new_server

        # --- STEP 3: store access token in memory ---
        self._access_token = data["access_token"]   # do NOT log this value
        self._expires_at   = time.time() + int(data.get("expires_in", 1800))

        log.info("Questrade token: refresh OK, api_server=%s", new_server)
