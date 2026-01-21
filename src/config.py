"""Configuration management for the Polymarket arbitrage bot."""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class WalletConfig:
    """Wallet configuration for trading."""

    private_key: str = field(default_factory=lambda: os.getenv("PK", ""))
    funder_address: str = field(default_factory=lambda: os.getenv("FUNDER", ""))

    def validate(self) -> bool:
        """Check if wallet is configured."""
        return bool(self.private_key and self.funder_address)


@dataclass
class APIConfig:
    """API credentials for Polymarket CLOB."""

    api_key: Optional[str] = field(default_factory=lambda: os.getenv("CLOB_API_KEY"))
    api_secret: Optional[str] = field(default_factory=lambda: os.getenv("CLOB_SECRET"))
    api_passphrase: Optional[str] = field(default_factory=lambda: os.getenv("CLOB_PASS_PHRASE"))

    @property
    def is_configured(self) -> bool:
        """Check if API credentials are set."""
        return all([self.api_key, self.api_secret, self.api_passphrase])


@dataclass
class TradingConfig:
    """Trading parameters and thresholds."""

    # Minimum edge required to enter a trade (as decimal, e.g., 0.25 = 25%)
    min_edge: float = field(
        default_factory=lambda: float(os.getenv("MIN_EDGE", "0.25"))
    )

    # Minimum time remaining before market close (seconds)
    min_time_remaining: float = field(
        default_factory=lambda: float(os.getenv("MIN_TIME_REMAINING", "45"))
    )

    # Base position size in USD
    base_position_size: float = field(
        default_factory=lambda: float(os.getenv("BASE_POSITION_SIZE", "50"))
    )

    # Maximum position as percentage of bankroll
    max_position_pct: float = field(
        default_factory=lambda: float(os.getenv("MAX_POSITION_PCT", "0.10"))
    )

    # Maximum concurrent positions
    max_concurrent_positions: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONCURRENT_POSITIONS", "5"))
    )

    # Daily loss limit as percentage of starting bankroll
    daily_loss_limit: float = field(
        default_factory=lambda: float(os.getenv("DAILY_LOSS_LIMIT", "0.20"))
    )

    # Edge thresholds for different order types (configurable via env)
    edge_for_post_only: float = field(
        default_factory=lambda: float(os.getenv("EDGE_FOR_POST_ONLY", "0.30"))
    )  # 30% edge -> POST_ONLY
    edge_for_limit: float = field(
        default_factory=lambda: float(os.getenv("EDGE_FOR_LIMIT", "0.40"))
    )  # 40% edge -> LIMIT (may take)
    edge_for_market: float = field(
        default_factory=lambda: float(os.getenv("EDGE_FOR_MARKET", "0.50"))
    )  # 50% edge -> MARKET (guaranteed fill)

    # Time thresholds for urgency (configurable via env)
    time_for_market: float = field(
        default_factory=lambda: float(os.getenv("TIME_FOR_MARKET", "60.0"))
    )  # < 60s remaining -> MARKET
    time_for_limit: float = field(
        default_factory=lambda: float(os.getenv("TIME_FOR_LIMIT", "120.0"))
    )  # < 120s remaining -> LIMIT

    # Order timeout for maker orders (seconds)
    maker_order_timeout: float = field(
        default_factory=lambda: float(os.getenv("MAKER_ORDER_TIMEOUT", "30.0"))
    )


@dataclass
class VolatilityConfig:
    """Default volatility assumptions for each asset (15-minute)."""

    # Conservative volatility estimates (as decimals)
    btc: float = 0.0035  # 0.35%
    eth: float = 0.0045  # 0.45%
    sol: float = 0.0070  # 0.70%
    xrp: float = 0.0060  # 0.60%

    def get(self, asset: str) -> float:
        """Get volatility for an asset."""
        asset_lower = asset.lower()
        return getattr(self, asset_lower, 0.005)  # Default 0.5%


@dataclass
class WebSocketConfig:
    """WebSocket connection and retry settings."""

    # Maximum consecutive reconnection attempts before circuit breaker opens
    max_retries: int = field(
        default_factory=lambda: int(os.getenv("WS_MAX_RETRIES", "10"))
    )

    # Initial reconnect delay (seconds)
    initial_reconnect_delay: float = field(
        default_factory=lambda: float(os.getenv("WS_RECONNECT_DELAY", "1.0"))
    )

    # Maximum reconnect delay (seconds)
    max_reconnect_delay: float = field(
        default_factory=lambda: float(os.getenv("WS_MAX_RECONNECT_DELAY", "60.0"))
    )

    # Ping interval for keepalive (seconds)
    ping_interval: int = field(
        default_factory=lambda: int(os.getenv("WS_PING_INTERVAL", "30"))
    )

    # Ping timeout (seconds)
    ping_timeout: int = field(
        default_factory=lambda: int(os.getenv("WS_PING_TIMEOUT", "10"))
    )


@dataclass
class EndpointsConfig:
    """API and WebSocket endpoints."""

    # Chainlink Real-Time Data Stream
    chainlink_rtds_url: str = "wss://ws-live-data.polymarket.com"
    chainlink_topic: str = "crypto_prices_chainlink"

    # Polymarket CLOB WebSocket
    clob_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    # Polymarket CLOB REST API
    clob_api_url: str = "https://clob.polymarket.com"

    # Gamma API for market discovery
    gamma_api_url: str = "https://gamma-api.polymarket.com"


@dataclass
class NotificationsConfig:
    """Optional notification settings."""

    telegram_bot_token: Optional[str] = field(
        default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN")
    )
    telegram_chat_id: Optional[str] = field(
        default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID")
    )
    discord_webhook_url: Optional[str] = field(
        default_factory=lambda: os.getenv("DISCORD_WEBHOOK_URL")
    )

    @property
    def telegram_enabled(self) -> bool:
        """Check if Telegram notifications are configured."""
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    @property
    def discord_enabled(self) -> bool:
        """Check if Discord notifications are configured."""
        return bool(self.discord_webhook_url)


@dataclass
class BotConfig:
    """Main configuration container for the trading bot."""

    wallet: WalletConfig = field(default_factory=WalletConfig)
    api: APIConfig = field(default_factory=APIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    endpoints: EndpointsConfig = field(default_factory=EndpointsConfig)
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)

    # Supported assets for 15-minute markets
    supported_assets: list = field(
        default_factory=lambda: ["BTC", "ETH", "SOL", "XRP"]
    )

    # Trading loop interval (seconds)
    loop_interval: float = 0.5

    # Market refresh interval (seconds)
    market_refresh_interval: float = 60.0

    # Log file path
    trades_log_file: str = "trades.jsonl"

    # Dry run mode (no actual trades)
    dry_run: bool = field(
        default_factory=lambda: os.getenv("DRY_RUN", "false").lower() == "true"
    )

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate the configuration.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        if not self.wallet.validate():
            errors.append("Wallet not configured: PK and FUNDER required")

        if self.trading.min_edge <= 0 or self.trading.min_edge >= 1:
            errors.append("min_edge must be between 0 and 1")

        if self.trading.min_time_remaining < 0:
            errors.append("min_time_remaining must be positive")

        if self.trading.base_position_size <= 0:
            errors.append("base_position_size must be positive")

        if self.trading.max_position_pct <= 0 or self.trading.max_position_pct > 1:
            errors.append("max_position_pct must be between 0 and 1")

        return (len(errors) == 0, errors)

    @classmethod
    def from_env(cls) -> "BotConfig":
        """Create configuration from environment variables."""
        return cls()

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for logging."""
        return {
            "trading": {
                "min_edge": self.trading.min_edge,
                "min_time_remaining": self.trading.min_time_remaining,
                "base_position_size": self.trading.base_position_size,
                "max_position_pct": self.trading.max_position_pct,
                "max_concurrent_positions": self.trading.max_concurrent_positions,
                "daily_loss_limit": self.trading.daily_loss_limit,
            },
            "volatility": {
                "btc": self.volatility.btc,
                "eth": self.volatility.eth,
                "sol": self.volatility.sol,
                "xrp": self.volatility.xrp,
            },
            "dry_run": self.dry_run,
            "supported_assets": self.supported_assets,
        }


# Global configuration instance
config = BotConfig.from_env()
