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

    @property
    def is_proxy_wallet(self) -> bool:
        """Check if using proxy wallet mode (browser/Magic wallet)."""
        return bool(self.funder_address)

    @property
    def signature_type(self) -> int:
        """Get signature type based on wallet configuration.

        Returns:
            0 for standard EOA wallet
            1 for proxy wallet (Magic/browser wallet with funder)
        """
        return 1 if self.is_proxy_wallet else 0

    def validate(self) -> bool:
        """Check if wallet is configured (at minimum, private key required)."""
        return bool(self.private_key)


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

    # Trading mode: "conservative", "normal", or "aggressive"
    # Aggressive mode enables compounding and Kelly-based sizing
    mode: str = field(
        default_factory=lambda: os.getenv("TRADING_MODE", "normal")
    )

    # Minimum edge required to enter a trade (as decimal, e.g., 0.05 = 5%)
    # Realistic values: 3-8% is typical for arbitrage opportunities
    min_edge: float = field(
        default_factory=lambda: float(os.getenv("MIN_EDGE", "0.03"))
    )

    # Minimum time remaining before market close (seconds)
    min_time_remaining: float = field(
        default_factory=lambda: float(os.getenv("MIN_TIME_REMAINING", "30"))
    )

    # Base position size in USD
    base_position_size: float = field(
        default_factory=lambda: float(os.getenv("BASE_POSITION_SIZE", "25"))
    )

    # Maximum position as percentage of bankroll
    max_position_pct: float = field(
        default_factory=lambda: float(os.getenv("MAX_POSITION_PCT", "0.15"))
    )

    # Maximum concurrent positions
    max_concurrent_positions: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONCURRENT_POSITIONS", "2"))
    )

    # Daily loss limit as percentage of starting bankroll
    daily_loss_limit: float = field(
        default_factory=lambda: float(os.getenv("DAILY_LOSS_LIMIT", "0.25"))
    )

    # === Advanced Loss Protection ===

    # Stop trading after X consecutive losses
    max_consecutive_losses: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3"))
    )

    # Stop trading if account drops X% from peak (drawdown protection)
    max_drawdown_pct: float = field(
        default_factory=lambda: float(os.getenv("MAX_DRAWDOWN_PCT", "0.15"))
    )

    # Pause trading if win rate drops below this (after min_trades_for_winrate)
    min_win_rate: float = field(
        default_factory=lambda: float(os.getenv("MIN_WIN_RATE", "0.40"))
    )

    # Minimum trades before win rate check kicks in
    min_trades_for_winrate: int = field(
        default_factory=lambda: int(os.getenv("MIN_TRADES_FOR_WINRATE", "5"))
    )

    # Cooling off period after hitting any limit (minutes)
    cooloff_period_minutes: int = field(
        default_factory=lambda: int(os.getenv("COOLOFF_PERIOD_MINUTES", "30"))
    )

    # === Machine Learning Settings ===

    # Enable ML signal filtering
    ml_enabled: bool = field(
        default_factory=lambda: os.getenv("ML_ENABLED", "true").lower() == "true"
    )

    # Minimum ML confidence to take a trade (0.5 = 50%)
    ml_min_confidence: float = field(
        default_factory=lambda: float(os.getenv("ML_MIN_CONFIDENCE", "0.55"))
    )

    # Minimum training samples before ML filtering kicks in
    ml_min_samples: int = field(
        default_factory=lambda: int(os.getenv("ML_MIN_SAMPLES", "10"))
    )

    # === Trade History Review Settings ===
    # Bot reviews past performance before taking new trades

    # Minimum completed trades before history filter kicks in
    history_min_trades: int = field(
        default_factory=lambda: int(os.getenv("HISTORY_MIN_TRADES", "5"))
    )

    # Minimum win rate for asset/side to continue trading
    history_min_win_rate: float = field(
        default_factory=lambda: float(os.getenv("HISTORY_MIN_WIN_RATE", "0.35"))
    )

    # Minimum prediction accuracy required (if ML predictions available)
    history_min_prediction_accuracy: float = field(
        default_factory=lambda: float(os.getenv("HISTORY_MIN_PREDICTION_ACCURACY", "0.45"))
    )

    # === Arbitrage Strategy Settings ===
    # Based on successful bot patterns that made $5-10k daily

    # Dump threshold - price drop % that triggers entry (15% default)
    arb_dump_threshold: float = field(
        default_factory=lambda: float(os.getenv("ARB_DUMP_THRESHOLD", "0.15"))
    )

    # Hedge sum threshold - execute hedge when leg1 + opposite <= this
    arb_hedge_threshold: float = field(
        default_factory=lambda: float(os.getenv("ARB_HEDGE_THRESHOLD", "0.95"))
    )

    # Entry window - focus on first N minutes of 15-minute period
    arb_entry_window_minutes: float = field(
        default_factory=lambda: float(os.getenv("ARB_ENTRY_WINDOW", "2.0"))
    )

    # Minimum spread profit after fees (2.5% default)
    arb_min_spread: float = field(
        default_factory=lambda: float(os.getenv("ARB_MIN_SPREAD", "0.025"))
    )

    # === Observation/Warm-up Settings ===

    # Warm-up period at startup - observe before trading (seconds)
    warmup_period_seconds: int = field(
        default_factory=lambda: int(os.getenv("WARMUP_PERIOD", "60"))
    )

    # Minimum price samples needed before trading an asset
    min_price_samples: int = field(
        default_factory=lambda: int(os.getenv("MIN_PRICE_SAMPLES", "10"))
    )

    # Minimum observation time per market before trading (seconds)
    min_observation_time: float = field(
        default_factory=lambda: float(os.getenv("MIN_OBSERVATION_TIME", "30"))
    )

    # Edge thresholds for different order types (configurable via env)
    # These are realistic thresholds for actual trading
    edge_for_post_only: float = field(
        default_factory=lambda: float(os.getenv("EDGE_FOR_POST_ONLY", "0.03"))
    )  # 3% edge -> POST_ONLY (maker orders, earn rebates)
    edge_for_limit: float = field(
        default_factory=lambda: float(os.getenv("EDGE_FOR_LIMIT", "0.08"))
    )  # 8% edge -> LIMIT (may take liquidity)
    edge_for_market: float = field(
        default_factory=lambda: float(os.getenv("EDGE_FOR_MARKET", "0.15"))
    )  # 15% edge -> MARKET (guaranteed fill, pays fees)

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
