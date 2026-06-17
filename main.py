#!/usr/bin/env python3
"""
Polymarket 15-Minute Crypto Arbitrage Bot

Entry point for the trading bot. Handles command-line arguments,
configuration loading, and bot lifecycle management.

Usage:
    python main.py                     # Run with default settings
    python main.py --dry-run           # Run without actual trades
    python main.py --log-level DEBUG   # Enable debug logging
"""

import argparse
import asyncio
import signal
import sys

from src.config import BotConfig
from src.bot import TradingBot
from src.utils.logging import setup_logging, log_bot_start, log_bot_stop
from src.utils.console import setup_colored_logging, print_banner


# Global bot instance for signal handling
bot: TradingBot = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Polymarket 15-Minute Crypto Arbitrage Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                     # Run with settings from .env
    python main.py --dry-run           # Simulate trading without real orders
    python main.py --log-level DEBUG   # Enable verbose logging

Environment Variables:
    PK                  - Private key for wallet
    FUNDER              - Deposit address
    CLOB_API_KEY        - API key (optional, derived if not set)
    CLOB_SECRET         - API secret (optional)
    CLOB_PASS_PHRASE    - API passphrase (optional)
    DRY_RUN             - Set to 'true' for simulation mode
    MIN_EDGE            - Minimum edge to trade (default: 0.25)
    BASE_POSITION_SIZE  - Position size in USD (default: 50)
        """,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in simulation mode without actual trades",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (optional)",
    )

    parser.add_argument(
        "--min-edge",
        type=float,
        default=None,
        help="Minimum edge to trade (0.0 - 1.0)",
    )

    parser.add_argument(
        "--position-size",
        type=float,
        default=None,
        help="Base position size in USD",
    )

    parser.add_argument(
        "--reset-drawdown",
        action="store_true",
        help="Reset drawdown tracking (set peak to current balance)",
    )

    return parser.parse_args()


def setup_signal_handlers():
    """Set up handlers for graceful shutdown."""
    def handle_shutdown(signum, frame):
        """Handle shutdown signal."""
        print("\nShutdown signal received...")
        if bot:
            asyncio.create_task(bot.shutdown())

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)


async def main():
    """Main entry point."""
    global bot

    # Parse arguments
    args = parse_args()

    # Print banner
    print_banner()

    # Set up colored logging
    import logging
    log_level = getattr(logging, args.log_level)
    setup_colored_logging(level=log_level)

    # Also set up file logging if specified
    if args.log_file:
        setup_logging(
            level=args.log_level,
            log_file=args.log_file,
        )

    # Load configuration
    config = BotConfig.from_env()

    # Override with command-line arguments
    if args.dry_run:
        config.dry_run = True

    if args.min_edge is not None:
        config.trading.min_edge = args.min_edge

    if args.position_size is not None:
        config.trading.base_position_size = args.position_size

    # Validate configuration
    is_valid, errors = config.validate()
    if not is_valid:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease check your .env file or command-line arguments.")
        sys.exit(1)

    # Log startup
    log_bot_start(config)

    # Set up signal handlers
    setup_signal_handlers()

    # Create and run bot
    bot = TradingBot(config)

    # Reset drawdown if requested. NOTE: this must happen AFTER state restore
    # (which runs inside bot.start() -> initialize()), otherwise the restore
    # immediately overwrites the reset. We set a flag the bot honors post-restore.
    if args.reset_drawdown:
        print("🔄 Drawdown reset requested — will apply after state restore...")
        bot.reset_drawdown_requested = True

    try:
        await bot.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        raise
    finally:
        log_bot_stop(config)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
