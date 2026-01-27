"""
Colorful console output utilities for the trading bot.

Provides beautiful, color-coded logging with emojis and formatting
for easy monitoring of bot activity.
"""

import logging
import sys
from datetime import datetime
from typing import Optional


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""

    # Reset
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Regular colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


class ColoredFormatter(logging.Formatter):
    """
    Custom log formatter with colors based on log level and content.
    """

    # Level colors
    LEVEL_COLORS = {
        logging.DEBUG: Colors.DIM + Colors.WHITE,
        logging.INFO: Colors.BRIGHT_WHITE,
        logging.WARNING: Colors.BRIGHT_YELLOW,
        logging.ERROR: Colors.BRIGHT_RED,
        logging.CRITICAL: Colors.BG_RED + Colors.BRIGHT_WHITE,
    }

    # Keywords to highlight
    KEYWORDS = {
        # Positive events (green)
        "WIN": Colors.BRIGHT_GREEN,
        "SUCCESS": Colors.BRIGHT_GREEN,
        "PROFIT": Colors.BRIGHT_GREEN,
        "CONNECTED": Colors.BRIGHT_GREEN,
        "OPENED": Colors.BRIGHT_GREEN,
        "EXECUTING": Colors.BRIGHT_GREEN,

        # Negative events (red)
        "LOSS": Colors.BRIGHT_RED,
        "FAILED": Colors.BRIGHT_RED,
        "ERROR": Colors.BRIGHT_RED,
        "REJECTED": Colors.BRIGHT_RED,
        "CLOSED": Colors.BRIGHT_RED,

        # Neutral/info events (cyan/blue)
        "NEW MARKET": Colors.BRIGHT_CYAN,
        "MARKET": Colors.CYAN,
        "EXPIRING": Colors.BRIGHT_YELLOW,
        "SETTLING": Colors.BRIGHT_MAGENTA,
        "RESOLVED": Colors.BRIGHT_MAGENTA,

        # Assets (bright colors)
        "BTC": Colors.BRIGHT_YELLOW,
        "ETH": Colors.BRIGHT_BLUE,
        "SOL": Colors.BRIGHT_MAGENTA,
        "XRP": Colors.BRIGHT_CYAN,

        # Sides
        "UP": Colors.BRIGHT_GREEN,
        "DOWN": Colors.BRIGHT_RED,

        # Modes
        "SIMULATION": Colors.BRIGHT_YELLOW,
        "LIVE TRADING": Colors.BRIGHT_RED + Colors.BOLD,
        "DRY RUN": Colors.BRIGHT_YELLOW,
    }

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        # Get base color for level
        level_color = self.LEVEL_COLORS.get(record.levelno, Colors.WHITE)

        # Format timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Format level name with color
        level_name = record.levelname
        if record.levelno == logging.DEBUG:
            level_str = f"{Colors.DIM}DEBUG{Colors.RESET}"
        elif record.levelno == logging.INFO:
            level_str = f"{Colors.BRIGHT_BLUE}INFO {Colors.RESET}"
        elif record.levelno == logging.WARNING:
            level_str = f"{Colors.BRIGHT_YELLOW}WARN {Colors.RESET}"
        elif record.levelno == logging.ERROR:
            level_str = f"{Colors.BRIGHT_RED}ERROR{Colors.RESET}"
        elif record.levelno == logging.CRITICAL:
            level_str = f"{Colors.BG_RED}{Colors.BRIGHT_WHITE}CRIT {Colors.RESET}"
        else:
            level_str = level_name

        # Format the message
        message = record.getMessage()

        # Apply keyword highlighting
        for keyword, color in self.KEYWORDS.items():
            if keyword in message:
                message = message.replace(keyword, f"{color}{keyword}{Colors.RESET}")

        # Highlight dollar amounts
        import re
        message = re.sub(
            r'\$([0-9,]+\.?[0-9]*)',
            f'{Colors.BRIGHT_GREEN}$\\1{Colors.RESET}',
            message
        )

        # Highlight percentages
        message = re.sub(
            r'(\d+\.?\d*%)',
            f'{Colors.BRIGHT_CYAN}\\1{Colors.RESET}',
            message
        )

        # Highlight P&L with appropriate color
        message = re.sub(
            r'P&L: \$\+([0-9,]+\.?[0-9]*)',
            f'{Colors.BRIGHT_GREEN}P&L: $+\\1{Colors.RESET}',
            message
        )
        message = re.sub(
            r'P&L: \$-([0-9,]+\.?[0-9]*)',
            f'{Colors.BRIGHT_RED}P&L: $-\\1{Colors.RESET}',
            message
        )

        # Format logger name (shortened)
        logger_name = record.name
        if logger_name.startswith("src."):
            logger_name = logger_name[4:]  # Remove "src." prefix
        logger_name = logger_name[:12].ljust(12)  # Truncate and pad

        # Build final output
        time_color = Colors.DIM

        return (
            f"{time_color}{timestamp}{Colors.RESET} │ "
            f"{level_str} │ "
            f"{Colors.DIM}{logger_name}{Colors.RESET} │ "
            f"{message}"
        )


def setup_colored_logging(level: int = logging.INFO):
    """
    Setup colored logging for the entire application.

    Args:
        level: Logging level (default INFO)
    """
    # Create handler for stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter())
    handler.setLevel(level)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    # Add our colored handler
    root_logger.addHandler(handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("websocket").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def print_banner():
    """Print a beautiful startup banner."""
    banner = f"""
{Colors.BRIGHT_CYAN}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  {Colors.BRIGHT_WHITE}██████╗  ██████╗ ██╗  ██╗   ██╗███╗   ███╗ █████╗ ██████╗ {Colors.BRIGHT_CYAN} ║
║  {Colors.BRIGHT_WHITE}██╔══██╗██╔═══██╗██║  ╚██╗ ██╔╝████╗ ████║██╔══██╗██╔══██╗{Colors.BRIGHT_CYAN} ║
║  {Colors.BRIGHT_WHITE}██████╔╝██║   ██║██║   ╚████╔╝ ██╔████╔██║███████║██████╔╝{Colors.BRIGHT_CYAN} ║
║  {Colors.BRIGHT_WHITE}██╔═══╝ ██║   ██║██║    ╚██╔╝  ██║╚██╔╝██║██╔══██║██╔══██╗{Colors.BRIGHT_CYAN} ║
║  {Colors.BRIGHT_WHITE}██║     ╚██████╔╝███████╗██║   ██║ ╚═╝ ██║██║  ██║██║  ██║{Colors.BRIGHT_CYAN} ║
║  {Colors.BRIGHT_WHITE}╚═╝      ╚═════╝ ╚══════╝╚═╝   ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝{Colors.BRIGHT_CYAN} ║
║                                                              ║
║  {Colors.BRIGHT_YELLOW}15-Minute Crypto Arbitrage Bot{Colors.BRIGHT_CYAN}                            ║
║  {Colors.DIM}Chainlink Oracle × Polymarket CLOB{Colors.BRIGHT_CYAN}                        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
    print(banner)


def print_status_box(
    mode: str,
    balance: Optional[float],
    open_orders: int,
    positions: int,
    daily_pnl: float = 0.0,
    win_rate: float = 0.0,
):
    """
    Print a formatted status box.

    Args:
        mode: Trading mode (SIMULATION or LIVE)
        balance: Account balance in USDC
        open_orders: Number of open orders
        positions: Number of open positions
        daily_pnl: Daily P&L
        win_rate: Win rate percentage
    """
    mode_color = Colors.BRIGHT_YELLOW if "SIM" in mode else Colors.BRIGHT_RED
    balance_str = f"${balance:,.2f}" if balance else "N/A"
    pnl_color = Colors.BRIGHT_GREEN if daily_pnl >= 0 else Colors.BRIGHT_RED
    pnl_sign = "+" if daily_pnl >= 0 else ""

    box = f"""
{Colors.BRIGHT_CYAN}┌────────────────────────────────────────────────────────────┐
│{Colors.RESET}  {Colors.BOLD}Account Status{Colors.RESET}                                           {Colors.BRIGHT_CYAN}│
├────────────────────────────────────────────────────────────┤
│{Colors.RESET}  Mode:        {mode_color}{mode:20}{Colors.RESET}                       {Colors.BRIGHT_CYAN}│
│{Colors.RESET}  Balance:     {Colors.BRIGHT_GREEN}{balance_str:20}{Colors.RESET}                       {Colors.BRIGHT_CYAN}│
│{Colors.RESET}  Positions:   {Colors.BRIGHT_WHITE}{positions:<20}{Colors.RESET}                       {Colors.BRIGHT_CYAN}│
│{Colors.RESET}  Open Orders: {Colors.BRIGHT_WHITE}{open_orders:<20}{Colors.RESET}                       {Colors.BRIGHT_CYAN}│
│{Colors.RESET}  Daily P&L:   {pnl_color}{pnl_sign}${abs(daily_pnl):,.2f}{Colors.RESET}                                      {Colors.BRIGHT_CYAN}│
│{Colors.RESET}  Win Rate:    {Colors.BRIGHT_CYAN}{win_rate:.1f}%{Colors.RESET}                                          {Colors.BRIGHT_CYAN}│
└────────────────────────────────────────────────────────────┘{Colors.RESET}
"""
    print(box)


def print_config_box(config_dict: dict):
    """
    Print a formatted configuration box.

    Args:
        config_dict: Dictionary of config key-value pairs
    """
    print(f"\n{Colors.BRIGHT_CYAN}┌─────────────────── Trading Config ───────────────────┐{Colors.RESET}")

    for key, value in config_dict.items():
        key_str = key.ljust(25)
        if isinstance(value, float):
            if value < 1:
                value_str = f"{value:.0%}"
            else:
                value_str = f"${value:,.2f}" if value > 10 else f"{value}"
        else:
            value_str = str(value)

        print(f"{Colors.BRIGHT_CYAN}│{Colors.RESET}  {Colors.DIM}{key_str}{Colors.RESET} {Colors.BRIGHT_WHITE}{value_str:>25}{Colors.RESET} {Colors.BRIGHT_CYAN}│{Colors.RESET}")

    print(f"{Colors.BRIGHT_CYAN}└──────────────────────────────────────────────────────┘{Colors.RESET}\n")
