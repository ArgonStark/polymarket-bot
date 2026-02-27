"""Configuration management for the Polymarket arbitrage bot."""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

_config_logger = logging.getLogger(__name__)


def _safe_float(env_var: str, default: str) -> float:
    """Parse float from env var with fallback on bad values."""
    raw = os.getenv(env_var, default)
    try:
        return float(raw)
    except (ValueError, TypeError):
        _config_logger.warning(
            "ENV %s=%r is not a valid float, using default %s", env_var, raw, default
        )
        return float(default)


def _safe_int(env_var: str, default: str) -> int:
    """Parse int from env var with fallback on bad values."""
    raw = os.getenv(env_var, default)
    try:
        return int(raw)
    except (ValueError, TypeError):
        _config_logger.warning(
            "ENV %s=%r is not a valid int, using default %s", env_var, raw, default
        )
        return int(default)


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

    # SIMPLE MODE (RECOMMENDED): Use distance-from-target strategy
    # This is the most profitable approach for 15-minute markets
    # Set SIMPLE_MODE=false to use complex indicators (not recommended)
    simple_mode: bool = field(
        default_factory=lambda: os.getenv("SIMPLE_MODE", "true").lower() == "true"
    )

    # AGGRESSIVE MODE: Trade every market, always pick a side
    # Set AGGRESSIVE_MODE=true to enable
    # This overrides normal signal generation with simple momentum-based decisions
    aggressive_mode: bool = field(
        default_factory=lambda: os.getenv("AGGRESSIVE_MODE", "false").lower() == "true"
    )

    # Minimum edge required to enter a trade (as decimal, e.g., 0.05 = 5%)
    # With SIMPLE_MODE, lower edge is OK because signals are more reliable
    # 0% = trade all signals, 2% = only trade when we have clear edge
    min_edge: float = field(
        default_factory=lambda: _safe_float("MIN_EDGE", "0.00")
    )

    # Minimum time remaining before market close (seconds)
    # 30s is enough for order execution - closer to settlement = more confident
    min_time_remaining: float = field(
        default_factory=lambda: _safe_float("MIN_TIME_REMAINING", "30")
    )

    # Base position size in USD (smaller = more trades, less risk per trade)
    base_position_size: float = field(
        default_factory=lambda: _safe_float("BASE_POSITION_SIZE", "25")
    )

    # Maximum position as percentage of bankroll
    max_position_pct: float = field(
        default_factory=lambda: _safe_float("MAX_POSITION_PCT", "0.10")
    )

    # Maximum concurrent positions (filled + pending orders)
    # 4 = one per asset (BTC, ETH, SOL, XRP)
    max_concurrent_positions: int = field(
        default_factory=lambda: _safe_int("MAX_CONCURRENT_POSITIONS", "4")
    )

    # Maximum order submits per rolling 60 seconds
    max_orders_per_min: int = field(
        default_factory=lambda: _safe_int("MAX_ORDERS_PER_MIN", "10")
    )

    # Absolute exposure caps in USD
    max_total_exposure_usd: float = field(
        default_factory=lambda: _safe_float("MAX_TOTAL_EXPOSURE_USD", "200")
    )
    max_exposure_per_asset_usd: float = field(
        default_factory=lambda: _safe_float("MAX_EXPOSURE_PER_ASSET_USD", "75")
    )

    # Maximum total capital at risk (as % of bankroll)
    # Prevents over-exposure even if individual position limits are met
    max_total_exposure_pct: float = field(
        default_factory=lambda: _safe_float("MAX_TOTAL_EXPOSURE_PCT", "0.25")
    )

    # Cooldown between trades for SAME asset (seconds)
    # 10s enough to prevent duplicates while allowing active trading
    order_cooldown_seconds: int = field(
        default_factory=lambda: _safe_int("ORDER_COOLDOWN_SECONDS", "10")
    )

    # Asset priority - higher priority assets get checked first
    # BTC and ETH are most liquid, SOL has good volume
    # XRP has lowest priority - less correlated with BTC
    asset_priority: list = field(
        default_factory=lambda: os.getenv("ASSET_PRIORITY", "BTC,ETH,SOL,XRP").split(",")
    )

    # Enable parallel market processing (faster but more API calls)
    parallel_execution: bool = field(
        default_factory=lambda: os.getenv("PARALLEL_EXECUTION", "true").lower() == "true"
    )

    # Daily loss limit as percentage of starting bankroll
    daily_loss_limit: float = field(
        default_factory=lambda: _safe_float("DAILY_LOSS_LIMIT", "0.25")
    )

    # === Advanced Loss Protection ===

    # Stop trading after X consecutive losses
    max_consecutive_losses: int = field(
        default_factory=lambda: _safe_int("MAX_CONSECUTIVE_LOSSES", "5")
    )

    # Stop trading if account drops X% from peak (drawdown protection)
    max_drawdown_pct: float = field(
        default_factory=lambda: _safe_float("MAX_DRAWDOWN_PCT", "0.40")
    )

    # Pause trading if win rate drops below this (after min_trades_for_winrate)
    min_win_rate: float = field(
        default_factory=lambda: _safe_float("MIN_WIN_RATE", "0.40")
    )

    # Minimum trades before win rate check kicks in
    min_trades_for_winrate: int = field(
        default_factory=lambda: _safe_int("MIN_TRADES_FOR_WINRATE", "5")
    )

    # Cooling off period after hitting any limit (minutes)
    cooloff_period_minutes: int = field(
        default_factory=lambda: _safe_int("COOLOFF_PERIOD_MINUTES", "10")
    )

    # Peak decay rate after drawdown cooloff expires (fraction of gap per hour)
    # E.g. 0.05 = close 5% of (peak - current) gap per hour
    # Set to 0 to disable (manual reset_drawdown() required)
    peak_decay_rate_per_hour: float = field(
        default_factory=lambda: _safe_float("PEAK_DECAY_RATE_PER_HOUR", "0.05")
    )

    # === Market Variant Settings ===
    # Which market durations to trade: "five", "fifteen", or both
    # Default: only fifteen (existing behaviour)
    trading_variants: list = field(
        default_factory=lambda: [
            v.strip()
            for v in os.getenv("TRADING_VARIANTS", "fifteen").split(",")
            if v.strip() in ("five", "fifteen")
        ] or ["fifteen"]
    )

    # Per-variant timing overrides (5-min defaults are tighter)
    min_time_remaining_5m: float = field(
        default_factory=lambda: _safe_float("MIN_TIME_REMAINING_5M", "15")
    )
    min_observation_time_5m: float = field(
        default_factory=lambda: _safe_float("MIN_OBSERVATION_TIME_5M", "10")
    )

    # === Trade History Review Settings ===
    # Bot reviews past performance before taking new trades

    # Minimum completed trades before history filter kicks in
    history_min_trades: int = field(
        default_factory=lambda: _safe_int("HISTORY_MIN_TRADES", "5")
    )

    # Minimum win rate for asset/side to continue trading
    history_min_win_rate: float = field(
        default_factory=lambda: _safe_float("HISTORY_MIN_WIN_RATE", "0.35")
    )

    # Minimum prediction accuracy required (if ML predictions available)
    history_min_prediction_accuracy: float = field(
        default_factory=lambda: _safe_float("HISTORY_MIN_PREDICTION_ACCURACY", "0.45")
    )

    # === Chart Filter Settings ===
    # Block trades that go against strong chart signals (bearish chart + UP trade = blocked)
    chart_filter_enabled: bool = field(
        default_factory=lambda: os.getenv("CHART_FILTER_ENABLED", "true").lower() == "true"
    )

    # === Early Exit Settings (Take-Profit / Stop-Loss) ===
    # Close positions early to lock in profits or limit losses

    # Enable early exit feature (disabled by default - experimental)
    early_exit_enabled: bool = field(
        default_factory=lambda: os.getenv("EARLY_EXIT_ENABLED", "false").lower() == "true"
    )

    # Take-profit threshold (e.g., 0.30 = close when +30% profit)
    take_profit_pct: float = field(
        default_factory=lambda: _safe_float("TAKE_PROFIT_PCT", "0.30")
    )

    # Stop-loss threshold (e.g., 0.25 = close when -25% loss)
    stop_loss_pct: float = field(
        default_factory=lambda: _safe_float("STOP_LOSS_PCT", "0.25")
    )

    # Minimum time into position before allowing early exit (seconds)
    # Prevents exiting too quickly on noise
    early_exit_min_hold_time: float = field(
        default_factory=lambda: _safe_float("EARLY_EXIT_MIN_HOLD", "60")
    )

    # Check positions for early exit every N seconds
    early_exit_check_interval: float = field(
        default_factory=lambda: _safe_float("EARLY_EXIT_INTERVAL", "5")
    )

    # === Time-Based Exit ===
    # Close positions before resolution if edge decays
    time_exit_enabled: bool = field(
        default_factory=lambda: os.getenv("TIME_EXIT_ENABLED", "true").lower() == "true"
    )
    time_exit_seconds: float = field(
        default_factory=lambda: _safe_float("TIME_EXIT_SECONDS", "90")
    )
    exit_edge_threshold: float = field(
        default_factory=lambda: _safe_float("EXIT_EDGE_THRESHOLD", "0.01")
    )

    # === Arbitrage Strategy Settings ===
    # Based on successful bot patterns that made $5-10k daily

    # Dump threshold - price drop % that triggers entry (15% default)
    arb_dump_threshold: float = field(
        default_factory=lambda: _safe_float("ARB_DUMP_THRESHOLD", "0.15")
    )

    # Hedge sum threshold - execute hedge when leg1 + opposite <= this
    arb_hedge_threshold: float = field(
        default_factory=lambda: _safe_float("ARB_HEDGE_THRESHOLD", "0.95")
    )

    # Entry window - focus on first N minutes of 15-minute period
    arb_entry_window_minutes: float = field(
        default_factory=lambda: _safe_float("ARB_ENTRY_WINDOW", "2.0")
    )

    # Minimum spread profit after fees (2.5% default)
    arb_min_spread: float = field(
        default_factory=lambda: _safe_float("ARB_MIN_SPREAD", "0.025")
    )

    # === Observation/Warm-up Settings ===

    # Warm-up period at startup - observe before trading (seconds)
    warmup_period_seconds: int = field(
        default_factory=lambda: _safe_int("WARMUP_PERIOD", "60")
    )

    # Minimum price samples needed before trading an asset
    min_price_samples: int = field(
        default_factory=lambda: _safe_int("MIN_PRICE_SAMPLES", "10")
    )

    # Minimum observation time per market before trading (seconds)
    min_observation_time: float = field(
        default_factory=lambda: _safe_float("MIN_OBSERVATION_TIME", "30")
    )

    # Edge thresholds for different order types (configurable via env)
    # These are realistic thresholds for actual trading
    edge_for_post_only: float = field(
        default_factory=lambda: _safe_float("EDGE_FOR_POST_ONLY", "0.03")
    )  # 3% edge -> POST_ONLY (maker orders, earn rebates)
    edge_for_limit: float = field(
        default_factory=lambda: _safe_float("EDGE_FOR_LIMIT", "0.08")
    )  # 8% edge -> LIMIT (may take liquidity)
    edge_for_market: float = field(
        default_factory=lambda: _safe_float("EDGE_FOR_MARKET", "0.15")
    )  # 15% edge -> MARKET (guaranteed fill, pays fees)

    # Time thresholds for urgency (configurable via env)
    time_for_market: float = field(
        default_factory=lambda: _safe_float("TIME_FOR_MARKET", "60.0")
    )  # < 60s remaining -> MARKET
    time_for_limit: float = field(
        default_factory=lambda: _safe_float("TIME_FOR_LIMIT", "120.0")
    )  # < 120s remaining -> LIMIT

    # Order timeout for maker orders (seconds)
    maker_order_timeout: float = field(
        default_factory=lambda: _safe_float("MAKER_ORDER_TIMEOUT", "30.0")
    )

    # === Market Data Sanity Settings ===

    # Maximum bid-ask spread before rejecting quotes (e.g. 0.50 = 50%)
    max_spread: float = field(
        default_factory=lambda: _safe_float("MAX_SPREAD", "0.50")
    )

    # Maximum quote age in seconds before treating as stale
    quote_staleness_seconds: float = field(
        default_factory=lambda: _safe_float("QUOTE_STALENESS_SECONDS", "60.0")
    )

    # Probability consistency tolerance: abs(yes_mid + no_mid - 1.0) must be < this
    prob_tolerance: float = field(
        default_factory=lambda: _safe_float("PROB_TOLERANCE", "0.05")
    )

    # Bankroll regime: observe-only when bankroll < min_trade * this multiplier
    min_bankroll_multiplier: float = field(
        default_factory=lambda: _safe_float("MIN_BANKROLL_MULTIPLIER", "2.0")
    )

    # Capital buffer (fraction of bankroll available for trading, e.g. 0.90 = 10% reserve)
    capital_buffer: float = field(
        default_factory=lambda: _safe_float("CAPITAL_BUFFER", "0.90")
    )

    # Rolling trade history window size for recent PnL tracking
    trade_history_size: int = field(
        default_factory=lambda: _safe_int("TRADE_HISTORY_SIZE", "20")
    )

    def get_timing(self, variant: str) -> dict:
        """Return timing constants scaled for the market variant.

        5-min markets use 1/3 of 15-min thresholds (proportional to duration).
        """
        if variant == "five":
            scale = 300.0 / 900.0  # 5min / 15min
            return {
                "time_for_market": self.time_for_market * scale,
                "time_for_limit": self.time_for_limit * scale,
                "maker_order_timeout": max(10.0, self.maker_order_timeout * scale),
                "early_exit_min_hold": max(10.0, self.early_exit_min_hold_time * scale),
                "time_exit_seconds": max(15.0, self.time_exit_seconds * scale),
            }
        return {
            "time_for_market": self.time_for_market,
            "time_for_limit": self.time_for_limit,
            "maker_order_timeout": self.maker_order_timeout,
            "early_exit_min_hold": self.early_exit_min_hold_time,
            "time_exit_seconds": self.time_exit_seconds,
        }


@dataclass
class StrategyConfig:
    """Multi-strategy framework settings."""

    multi_strategy_enabled: bool = field(
        default_factory=lambda: os.getenv("MULTI_STRATEGY_ENABLED", "false").lower() == "true"
    )
    trend_weight: float = field(
        default_factory=lambda: _safe_float("STRATEGY_WEIGHT_TREND", "0.4")
    )
    mean_reversion_weight: float = field(
        default_factory=lambda: _safe_float("STRATEGY_WEIGHT_MEAN_REVERSION", "0.3")
    )
    momentum_weight: float = field(
        default_factory=lambda: _safe_float("STRATEGY_WEIGHT_MOMENTUM", "0.3")
    )


@dataclass
class RiskSizingConfig:
    """Dynamic risk sizing parameters."""

    enabled: bool = field(
        default_factory=lambda: os.getenv("DYNAMIC_SIZING_ENABLED", "true").lower() == "true"
    )
    target_volatility: float = field(
        default_factory=lambda: _safe_float("TARGET_VOLATILITY", "0.006")
    )
    kelly_cap: float = field(
        default_factory=lambda: _safe_float("KELLY_CAP", "0.20")
    )
    max_exposure_pct: float = field(
        default_factory=lambda: _safe_float("MAX_EXPOSURE_PCT", "0.10")
    )
    min_trade_usd: float = field(
        default_factory=lambda: _safe_float("MIN_TRADE_USD", "2.50")
    )


@dataclass
class ExecutionConfig:
    """Low-latency execution settings."""

    smart_router_enabled: bool = field(
        default_factory=lambda: os.getenv("SMART_ROUTER_ENABLED", "true").lower() == "true"
    )
    maker_edge_threshold: float = field(
        default_factory=lambda: _safe_float("MAKER_EDGE_THRESHOLD", "0.02")
    )
    taker_edge_threshold: float = field(
        default_factory=lambda: _safe_float("TAKER_EDGE_THRESHOLD", "0.06")
    )
    max_spread: float = field(
        default_factory=lambda: _safe_float("MAX_SPREAD", "0.05")
    )
    stale_order_seconds: float = field(
        default_factory=lambda: _safe_float("STALE_ORDER_SECONDS", "30")
    )


@dataclass
class PaperTradingConfig:
    """Paper trading settings."""

    enabled: bool = field(
        default_factory=lambda: os.getenv("PAPER_TRADING_ENABLED", "false").lower() == "true"
    )
    initial_balance: float = field(
        default_factory=lambda: _safe_float("PAPER_INITIAL_BALANCE", "1000")
    )
    maker_fee_bps: float = field(
        default_factory=lambda: _safe_float("PAPER_MAKER_FEE_BPS", "1.0")
    )
    taker_fee_bps: float = field(
        default_factory=lambda: _safe_float("PAPER_TAKER_FEE_BPS", "2.5")
    )
    slippage_bps: float = field(
        default_factory=lambda: _safe_float("PAPER_SLIPPAGE_BPS", "2.0")
    )
    state_file: str = field(
        default_factory=lambda: os.getenv("BOT_STATE_FILE", "bot_state.json")
    )


@dataclass
class MonitoringConfig:
    """Monitoring output settings."""

    enabled: bool = field(
        default_factory=lambda: os.getenv("MONITORING_ENABLED", "true").lower() == "true"
    )
    output_path: str = field(
        default_factory=lambda: os.getenv("MONITORING_OUTPUT_PATH", "data/metrics.jsonl")
    )
    rolling_window: int = field(
        default_factory=lambda: _safe_int("MONITORING_ROLLING_WINDOW", "50")
    )
    ws_port: int = field(
        default_factory=lambda: _safe_int("MONITOR_WS_PORT", "8765")
    )


@dataclass
class VolatilityConfig:
    """Default volatility assumptions for each asset (15-minute)."""

    # Volatility estimates (as decimals) - adjusted for more balanced trading
    btc: float = 0.0050  # 0.50% (was 0.35%)
    eth: float = 0.0055  # 0.55% (was 0.45%)
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
        default_factory=lambda: _safe_int("WS_MAX_RETRIES", "10")
    )

    # Initial reconnect delay (seconds)
    initial_reconnect_delay: float = field(
        default_factory=lambda: _safe_float("WS_RECONNECT_DELAY", "1.0")
    )

    # Maximum reconnect delay (seconds)
    max_reconnect_delay: float = field(
        default_factory=lambda: _safe_float("WS_MAX_RECONNECT_DELAY", "60.0")
    )

    # Ping interval for keepalive (seconds)
    ping_interval: int = field(
        default_factory=lambda: _safe_int("WS_PING_INTERVAL", "30")
    )

    # Ping timeout (seconds)
    ping_timeout: int = field(
        default_factory=lambda: _safe_int("WS_PING_TIMEOUT", "10")
    )


@dataclass
class TrendProtectionConfig:
    """Trend protection settings to avoid trading against strong market trends."""

    # Master enable/disable switch
    # DISABLED by default - let bot trade freely initially
    enabled: bool = field(
        default_factory=lambda: os.getenv("TREND_PROTECTION_ENABLED", "false").lower() == "true"
    )

    # === Trend Alignment Filter ===
    # Block trades against strong trends
    max_opposite_trend_1d: float = field(
        default_factory=lambda: _safe_float("MAX_OPPOSITE_TREND_1D", "0.25")
    )  # Block if 1d trend > 25% against signal

    max_opposite_trend_1h: float = field(
        default_factory=lambda: _safe_float("MAX_OPPOSITE_TREND_1H", "0.40")
    )  # Block if 1h trend > 40% against signal

    # === Price Velocity Guard ===
    # Skip trades when price moving too fast
    velocity_guard_enabled: bool = field(
        default_factory=lambda: os.getenv("VELOCITY_GUARD_ENABLED", "true").lower() == "true"
    )

    # Max velocity per asset (% per second) - skip if exceeded
    # Higher = more lenient = more trades allowed
    velocity_btc: float = field(
        default_factory=lambda: _safe_float("VELOCITY_MAX_BTC", "0.00035")
    )  # ~2.1% per minute (raised to trade more)

    velocity_eth: float = field(
        default_factory=lambda: _safe_float("VELOCITY_MAX_ETH", "0.00038")
    )  # ~2.3% per minute (raised to trade more)

    velocity_sol: float = field(
        default_factory=lambda: _safe_float("VELOCITY_MAX_SOL", "0.00030")
    )  # ~1.8% per minute

    velocity_xrp: float = field(
        default_factory=lambda: _safe_float("VELOCITY_MAX_XRP", "0.00025")
    )  # ~1.5% per minute (lowered to trade less)

    # === Multi-Timeframe Agreement ===
    timeframe_agreement_enabled: bool = field(
        default_factory=lambda: os.getenv("TIMEFRAME_AGREEMENT_ENABLED", "true").lower() == "true"
    )

    edge_boost_all_aligned: float = field(
        default_factory=lambda: _safe_float("EDGE_BOOST_ALL_ALIGNED", "0.01")
    )  # +1% when all 3 timeframes agree

    edge_penalty_mixed: float = field(
        default_factory=lambda: _safe_float("EDGE_PENALTY_MIXED", "0.01")
    )  # -1% when signals mixed

    # === Binance Momentum ===
    binance_momentum_enabled: bool = field(
        default_factory=lambda: os.getenv("BINANCE_MOMENTUM_ENABLED", "true").lower() == "true"
    )

    binance_velocity_threshold: float = field(
        default_factory=lambda: _safe_float("BINANCE_VELOCITY_THRESHOLD", "0.001")
    )  # 0.1% per second

    binance_momentum_boost: float = field(
        default_factory=lambda: _safe_float("BINANCE_MOMENTUM_BOOST", "0.02")
    )  # +2% edge boost for strong momentum

    # === Dynamic Edge Requirement ===
    dynamic_edge_enabled: bool = field(
        default_factory=lambda: os.getenv("DYNAMIC_EDGE_ENABLED", "true").lower() == "true"
    )

    dynamic_edge_min: float = field(
        default_factory=lambda: _safe_float("DYNAMIC_EDGE_MIN", "0.015")
    )  # Never go below 1.5%

    dynamic_edge_max: float = field(
        default_factory=lambda: _safe_float("DYNAMIC_EDGE_MAX", "0.05")
    )  # Never exceed 5%

    # === Consecutive Move Detection ===
    consecutive_enabled: bool = field(
        default_factory=lambda: os.getenv("CONSECUTIVE_ENABLED", "true").lower() == "true"
    )

    consecutive_min_moves: int = field(
        default_factory=lambda: _safe_int("CONSECUTIVE_MIN_MOVES", "3")
    )  # Minimum moves to trigger boost

    consecutive_boost_max: float = field(
        default_factory=lambda: _safe_float("CONSECUTIVE_BOOST_MAX", "0.02")
    )  # Max +2% edge boost

    def get_max_velocity(self, asset: str) -> float:
        """Get maximum allowed velocity for an asset."""
        asset_lower = asset.lower()
        return getattr(self, f"velocity_{asset_lower}", 0.0003)


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

    # Binance WebSocket for fast price updates (leading indicator)
    binance_ws_url: str = "wss://stream.binance.com:9443"

    # Use direct Binance WebSocket connection (requires external network access)
    # If false, uses Binance prices bundled via Polymarket's WebSocket (recommended)
    binance_direct_enabled: bool = field(
        default_factory=lambda: os.getenv("BINANCE_DIRECT", "false").lower() == "true"
    )


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
class SettlementConfig:
    """Settlement resolution thresholds and timing."""

    # CLOB bid threshold to declare winner (winning side >= this)
    clob_winner_threshold: float = field(
        default_factory=lambda: _safe_float("SETTLE_CLOB_WINNER_THRESHOLD", "0.85")
    )

    # CLOB bid threshold for losing side (<= this)
    clob_loser_threshold: float = field(
        default_factory=lambda: _safe_float("SETTLE_CLOB_LOSER_THRESHOLD", "0.15")
    )

    # Seconds after expiry before attempting local resolution
    local_delay_seconds: float = field(
        default_factory=lambda: _safe_float("SETTLE_LOCAL_DELAY", "30")
    )

    # Max seconds to wait for resolution before force-closing
    timeout_seconds: float = field(
        default_factory=lambda: _safe_float("SETTLE_TIMEOUT", "300")
    )

    # Seconds past end_time before settlement kicks in
    trigger_buffer: float = field(
        default_factory=lambda: _safe_float("SETTLE_TRIGGER_BUFFER", "2.0")
    )

    # Minimum margin (fraction) for local Chainlink resolution
    local_margin_min: float = field(
        default_factory=lambda: _safe_float("SETTLE_LOCAL_MARGIN_MIN", "0.001")
    )


@dataclass
class OFIConfig:
    """Order Flow Imbalance configuration (Binance aggTrade)."""

    enabled: bool = field(
        default_factory=lambda: os.getenv("OFI_ENABLED", "true").lower() == "true"
    )
    default_window: float = field(
        default_factory=lambda: _safe_float("OFI_WINDOW", "30")
    )
    min_ofi_threshold: float = field(
        default_factory=lambda: _safe_float("OFI_MIN_THRESHOLD", "0.15")
    )
    strong_ofi_threshold: float = field(
        default_factory=lambda: _safe_float("OFI_STRONG_THRESHOLD", "0.25")
    )
    confirmation_threshold: float = field(
        default_factory=lambda: _safe_float("OFI_CONFIRMATION_THRESHOLD", "0.10")
    )


@dataclass
class RegimeConfig:
    """Regime detection configuration (ATR + Efficiency Ratio)."""

    enabled: bool = field(
        default_factory=lambda: os.getenv("REGIME_ENABLED", "true").lower() == "true"
    )
    er_trending_threshold: float = field(
        default_factory=lambda: _safe_float("REGIME_ER_TRENDING", "0.60")
    )
    er_ranging_threshold: float = field(
        default_factory=lambda: _safe_float("REGIME_ER_RANGING", "0.35")
    )
    atr_low_percentile: float = field(
        default_factory=lambda: _safe_float("REGIME_ATR_LOW_PCT", "20")
    )
    ranging_kelly_mult: float = field(
        default_factory=lambda: _safe_float("REGIME_RANGING_KELLY_MULT", "0.50")
    )
    cache_ttl: float = field(
        default_factory=lambda: _safe_float("REGIME_CACHE_TTL", "5.0")
    )


@dataclass
class EdgeSignalConfig:
    """Unified edge signal configuration (Distance + OFI + Regime)."""

    enabled: bool = field(
        default_factory=lambda: os.getenv("EDGE_SIGNAL_ENABLED", "true").lower() == "true"
    )
    high_distance_threshold: float = field(
        default_factory=lambda: _safe_float("EDGE_HIGH_DISTANCE", "0.003")
    )
    medium_distance_threshold: float = field(
        default_factory=lambda: _safe_float("EDGE_MEDIUM_DISTANCE", "0.002")
    )
    complete_set_min_gap: float = field(
        default_factory=lambda: _safe_float("EDGE_COMPLETE_SET_GAP", "0.015")
    )
    kelly_high: float = field(
        default_factory=lambda: _safe_float("EDGE_KELLY_HIGH", "0.70")
    )
    kelly_medium: float = field(
        default_factory=lambda: _safe_float("EDGE_KELLY_MEDIUM", "0.40")
    )
    kelly_low: float = field(
        default_factory=lambda: _safe_float("EDGE_KELLY_LOW", "0.25")
    )
    max_edge: float = field(
        default_factory=lambda: _safe_float("EDGE_MAX", "0.12")
    )


@dataclass
class BotConfig:
    """Main configuration container for the trading bot."""

    wallet: WalletConfig = field(default_factory=WalletConfig)
    api: APIConfig = field(default_factory=APIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    trend_protection: TrendProtectionConfig = field(default_factory=TrendProtectionConfig)
    endpoints: EndpointsConfig = field(default_factory=EndpointsConfig)
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk_sizing: RiskSizingConfig = field(default_factory=RiskSizingConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    paper_trading: PaperTradingConfig = field(default_factory=PaperTradingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    settlement: SettlementConfig = field(default_factory=SettlementConfig)
    ofi: OFIConfig = field(default_factory=OFIConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    edge_signal: EdgeSignalConfig = field(default_factory=EdgeSignalConfig)

    # Supported assets for crypto up/down markets
    supported_assets: list = field(
        default_factory=lambda: ["BTC", "ETH", "SOL", "XRP"]
    )

    # Trading loop interval (seconds) - lower = faster signal detection
    loop_interval: float = field(
        default_factory=lambda: _safe_float("LOOP_INTERVAL", "0.25")
    )

    # Market refresh interval (seconds)
    market_refresh_interval: float = 60.0

    # Log file path
    trades_log_file: str = "trades.jsonl"

    # Dry run mode (no actual trades) — defaults to TRUE for safety.
    # Set DRY_RUN=false explicitly to enable live trading.
    dry_run: bool = field(
        default_factory=lambda: os.getenv("DRY_RUN", "true").lower() != "false"
    )

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate the configuration.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        warnings = []
        t = self.trading

        # --- Required credentials for live trading ---
        is_live = not self.dry_run and not self.paper_trading.enabled
        if is_live:
            if not self.wallet.validate():
                errors.append("Live trading requires PK env var (private key)")
            if not self.api.is_configured:
                warnings.append(
                    "No API credentials (CLOB_API_KEY/SECRET/PASS_PHRASE). "
                    "Client will attempt to derive them at startup."
                )

        # --- Trading parameter ranges ---
        if t.min_edge < 0 or t.min_edge >= 1:
            errors.append("MIN_EDGE must be in [0, 1)")
        if t.min_time_remaining < 0:
            errors.append("MIN_TIME_REMAINING must be >= 0")
        if t.base_position_size <= 0:
            errors.append("BASE_POSITION_SIZE must be > 0")
        if t.max_position_pct <= 0 or t.max_position_pct > 1:
            errors.append("MAX_POSITION_PCT must be in (0, 1]")
        if t.max_concurrent_positions < 1:
            errors.append("MAX_CONCURRENT_POSITIONS must be >= 1")
        if t.max_orders_per_min < 1:
            errors.append("MAX_ORDERS_PER_MIN must be >= 1")
        if t.max_total_exposure_usd <= 0:
            errors.append("MAX_TOTAL_EXPOSURE_USD must be > 0")
        if t.max_total_exposure_pct <= 0 or t.max_total_exposure_pct > 1:
            errors.append("MAX_TOTAL_EXPOSURE_PCT must be in (0, 1]")
        if t.daily_loss_limit <= 0 or t.daily_loss_limit > 1:
            errors.append("DAILY_LOSS_LIMIT must be in (0, 1]")
        if t.max_drawdown_pct <= 0 or t.max_drawdown_pct > 1:
            errors.append("MAX_DRAWDOWN_PCT must be in (0, 1]")
        if t.max_consecutive_losses < 1:
            errors.append("MAX_CONSECUTIVE_LOSSES must be >= 1")

        # --- Edge ordering: post_only < limit < market ---
        if not (t.edge_for_post_only <= t.edge_for_limit <= t.edge_for_market):
            warnings.append(
                f"Edge thresholds should be ordered: post_only({t.edge_for_post_only}) "
                f"<= limit({t.edge_for_limit}) <= market({t.edge_for_market})"
            )

        # --- Time ordering: time_for_market < time_for_limit ---
        if t.time_for_market >= t.time_for_limit:
            warnings.append(
                f"TIME_FOR_MARKET({t.time_for_market}s) should be < "
                f"TIME_FOR_LIMIT({t.time_for_limit}s)"
            )

        # --- Kelly / risk sizing ---
        rs = self.risk_sizing
        if rs.kelly_cap <= 0 or rs.kelly_cap > 1:
            errors.append("KELLY_CAP must be in (0, 1]")
        if rs.min_trade_usd <= 0:
            errors.append("MIN_TRADE_USD must be > 0")

        # --- Paper trading ---
        if self.paper_trading.enabled and self.paper_trading.initial_balance <= 0:
            errors.append("PAPER_INITIAL_BALANCE must be > 0")

        # --- WebSocket ---
        if self.websocket.max_retries < 1:
            errors.append("WS_MAX_RETRIES must be >= 1")

        # --- Loop interval ---
        if self.loop_interval <= 0:
            errors.append("LOOP_INTERVAL must be > 0")
        if self.loop_interval < 0.1:
            warnings.append("LOOP_INTERVAL < 0.1s may cause excessive CPU usage")

        # --- Log warnings ---
        for w in warnings:
            _config_logger.warning("Config warning: %s", w)

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
