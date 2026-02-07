"""CLI entrypoint for bot."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from src.bot import TradingBot
from src.config import BotConfig
from src.utils.logging import setup_logging, log_bot_start, log_bot_stop
from src.utils.console import setup_colored_logging, print_banner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml-bundle", "--ml_bundle", dest="ml_bundle", type=str, default=None)
    parser.add_argument("--min-edge", type=float, default=0.02)
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--log-file", type=str, default=None)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    if args.ml_bundle:
        os.environ["ML_ENGINE_ENABLED"] = "true"
        os.environ["ML_BUNDLE_PATH"] = args.ml_bundle
        os.environ["ML_BUNDLE"] = args.ml_bundle
    if args.min_edge is not None:
        os.environ["ML_MIN_EDGE"] = str(args.min_edge)
    if args.paper:
        os.environ["PAPER_TRADING_ENABLED"] = "true"
    if args.dry_run:
        os.environ["DRY_RUN"] = "true"

    print_banner()

    import logging
    log_level = getattr(logging, args.log_level)
    setup_colored_logging(level=log_level)

    if args.log_file:
        setup_logging(level=args.log_level, log_file=args.log_file)

    config = BotConfig.from_env()
    is_valid, errors = config.validate()
    if not is_valid:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    log_bot_start(config)
    bot = TradingBot(config)

    try:
        await bot.start()
    finally:
        log_bot_stop(config)


if __name__ == "__main__":
    asyncio.run(main())
