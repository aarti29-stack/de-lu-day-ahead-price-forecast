"""I run a basic smoke test for project configuration."""

from __future__ import annotations

from src.config import get_entsoe_api_key


def main() -> None:
    """I verify that configuration loading works end to end."""
    _ = get_entsoe_api_key()
    print("Config OK")


if __name__ == "__main__":
    main()
