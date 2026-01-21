"""
Logging utilities for the trading bot.

Provides structured logging for trades, signals, and bot events.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import requests

from ..models import Signal, TradeResult
from ..config import BotConfig


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging configuration for the bot.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output

    Returns:
        Root logger
    """
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Reduce noise from libraries
    logging.getLogger("websocket").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    return root_logger


def log_trade(
    signal: Signal,
    result: TradeResult,
    config: BotConfig,
):
    """
    Log a completed trade to file and optional notifications.

    Args:
        signal: Signal that generated the trade
        result: Result of the trade execution
        config: Bot configuration
    """
    logger = logging.getLogger(__name__)

    # Build trade entry
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signal_id": signal.signal_id,
        "market": {
            "condition_id": signal.market.condition_id,
            "asset": signal.market.asset,
            "target_price": signal.market.target_price,
            "question": signal.market.question[:100],
        },
        "signal": {
            "side": signal.side.value,
            "edge": signal.edge,
            "true_prob": signal.true_prob,
            "market_prob": signal.market_prob,
            "action": signal.recommended_action.value,
            "chainlink_price": signal.chainlink_price,
            "time_remaining": signal.time_remaining,
            "reasoning": signal.reasoning,
        },
        "order": {
            "size_usd": signal.size_usd,
            "size_shares": signal.size_shares,
            "recommended_price": signal.recommended_price,
        },
        "result": {
            "success": result.success,
            "order_id": result.order_id,
            "filled_size": result.filled_size,
            "filled_price": result.filled_price,
            "fee_paid": result.fee_paid,
            "error_message": result.error_message,
        },
    }

    # Write to JSONL file
    try:
        with open(config.trades_log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to write trade log: {e}")

    # Send notifications if configured
    if config.notifications.telegram_enabled:
        _send_telegram_notification(entry, config)

    if config.notifications.discord_enabled:
        _send_discord_notification(entry, config)


def _send_telegram_notification(entry: dict, config: BotConfig):
    """Send trade notification to Telegram."""
    logger = logging.getLogger(__name__)

    try:
        token = config.notifications.telegram_bot_token
        chat_id = config.notifications.telegram_chat_id

        signal = entry["signal"]
        result = entry["result"]
        market = entry["market"]

        status = "SUCCESS" if result["success"] else "FAILED"
        emoji = "+" if result["success"] else "X"

        message = (
            f"{emoji} **{status}**: {signal['side']} {market['asset']}\n"
            f"Edge: {signal['edge']:.1%} | "
            f"Price: ${signal['chainlink_price']:,.2f}\n"
            f"Size: ${entry['order']['size_usd']:.2f} | "
            f"Time: {signal['time_remaining']:.0f}s\n"
            f"Reasoning: {signal['reasoning'][:100]}"
        )

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown",
            },
            timeout=5,
        )

    except Exception as e:
        logger.error(f"Failed to send Telegram notification: {e}")


def _send_discord_notification(entry: dict, config: BotConfig):
    """Send trade notification to Discord webhook."""
    logger = logging.getLogger(__name__)

    try:
        webhook_url = config.notifications.discord_webhook_url

        signal = entry["signal"]
        result = entry["result"]
        market = entry["market"]

        color = 0x00FF00 if result["success"] else 0xFF0000
        status = "SUCCESS" if result["success"] else "FAILED"

        embed = {
            "title": f"{status}: {signal['side']} {market['asset']}",
            "color": color,
            "fields": [
                {
                    "name": "Edge",
                    "value": f"{signal['edge']:.1%}",
                    "inline": True,
                },
                {
                    "name": "Chainlink Price",
                    "value": f"${signal['chainlink_price']:,.2f}",
                    "inline": True,
                },
                {
                    "name": "Size",
                    "value": f"${entry['order']['size_usd']:.2f}",
                    "inline": True,
                },
                {
                    "name": "Time Remaining",
                    "value": f"{signal['time_remaining']:.0f}s",
                    "inline": True,
                },
                {
                    "name": "Reasoning",
                    "value": signal["reasoning"][:200],
                    "inline": False,
                },
            ],
            "timestamp": entry["timestamp"],
        }

        requests.post(
            webhook_url,
            json={"embeds": [embed]},
            timeout=5,
        )

    except Exception as e:
        logger.error(f"Failed to send Discord notification: {e}")


def log_bot_start(config: BotConfig):
    """Log bot startup."""
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("POLYMARKET 15-MIN CRYPTO ARBITRAGE BOT")
    logger.info("=" * 60)
    logger.info(f"Mode: {'DRY RUN' if config.dry_run else 'LIVE TRADING'}")
    logger.info(f"Min Edge: {config.trading.min_edge:.1%}")
    logger.info(f"Base Position Size: ${config.trading.base_position_size:.2f}")
    logger.info(f"Max Positions: {config.trading.max_concurrent_positions}")
    logger.info(f"Daily Loss Limit: {config.trading.daily_loss_limit:.1%}")
    logger.info(f"Supported Assets: {', '.join(config.supported_assets)}")
    logger.info("=" * 60)


def log_bot_stop(config: BotConfig):
    """Log bot shutdown."""
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("BOT SHUTDOWN")
    logger.info("=" * 60)
