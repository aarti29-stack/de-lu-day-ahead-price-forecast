"""I keep project configuration loading in one place."""

from __future__ import annotations

import os

from dotenv import load_dotenv


def get_entsoe_api_key() -> str:
    """I read ENTSO-E API credentials from the local environment.

    I load variables from .env and fail fast with a clear message when
    ENTSOE_API_KEY is empty or missing.
    """
    load_dotenv()
    api_key = os.getenv("ENTSOE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing ENTSOE_API_KEY. Add it to your .env file (see .env.example)."
        )
    return api_key
