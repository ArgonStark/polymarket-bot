"""Data models for the Polymarket arbitrage bot."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class OrderAction(Enum):
    """Recommended order action based on edge and time."""

    SKIP = "SKIP"           # No trade
    POST_ONLY = "POST_ONLY" # Maker order, wait for fill
    LIMIT = "LIMIT"         # GTC limit order (may take)
    MARKET = "MARKET"       # FOK market order (guaranteed fill)


class Side(Enum):
    """Trading side for prediction market."""

    UP = "UP"     # Price will be >= target
    DOWN = "DOWN" # Price will be < target
    NONE = "NONE" # No trade


class OrderStatus(Enum):
    """Status of an order."""

    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"


class PositionState(Enum):
    """
    State machine for position lifecycle.

    States:
    - OPEN: Market is active, position can be traded
    - ENDED_UNRESOLVED: Market ended but outcome not yet determined
    - RESOLVED_WIN: Outcome known, position wins (mark = 1.0)
    - RESOLVED_LOSS: Outcome known, position loses (mark = 0.0)
    - SETTLING: Redemption/cash settlement in progress
    - CLOSED: Finalized, realized PnL booked, position archived

    Important: SETTLING does NOT imply mark=1.0. Only RESOLVED_WIN does.
    """

    OPEN = "OPEN"
    ENDED_UNRESOLVED = "ENDED_UNRESOLVED"
    RESOLVED_WIN = "RESOLVED_WIN"
    RESOLVED_LOSS = "RESOLVED_LOSS"
    SETTLING = "SETTLING"
    CLOSED = "CLOSED"


@dataclass
class ChainlinkPrice:
    """Real-time price from Chainlink oracle."""

    symbol: str          # e.g., "btc/usd"
    price: float         # Current price
    timestamp: datetime  # Price timestamp

    @property
    def asset(self) -> str:
        """Extract asset symbol (e.g., 'BTC' from 'btc/usd')."""
        return self.symbol.split("/")[0].upper()


@dataclass
class OrderBookLevel:
    """Single level in the order book."""

    price: float  # Price (0.01 - 0.99)
    size: float   # Size in shares


@dataclass
class OrderBook:
    """Order book state for a token."""

    token_id: str
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def best_bid(self) -> Optional[float]:
        """Get best bid price."""
        if not self.bids:
            return None
        return max(level.price for level in self.bids)

    @property
    def best_ask(self) -> Optional[float]:
        """Get best ask price."""
        if not self.asks:
            return None
        return min(level.price for level in self.asks)

    @property
    def mid_price(self) -> Optional[float]:
        """Get mid price."""
        bid = self.best_bid
        ask = self.best_ask
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2

    @property
    def spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        bid = self.best_bid
        ask = self.best_ask
        if bid is None or ask is None:
            return None
        return ask - bid


@dataclass
class MarketState:
    """State of a 15-minute prediction market."""

    # Market identifiers
    condition_id: str           # Market condition ID
    question: str               # Market question text

    # Token IDs for trading
    up_token_id: str            # Token for UP outcome
    down_token_id: str          # Token for DOWN outcome

    # Asset info
    asset: str                  # "BTC", "ETH", "SOL", "XRP"
    target_price: float         # Start price for settlement

    # Timing
    start_time: datetime        # Market start time
    end_time: datetime          # Settlement time

    # Order book state (for UP token)
    best_bid: float = 0.0       # Best bid for UP
    best_ask: float = 1.0       # Best ask for UP
    bid_depth: float = 0.0      # Total bid depth
    ask_depth: float = 0.0      # Total ask depth

    # Metadata
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Settlement snapshot: Chainlink price captured when market moves to expiring queue
    expiry_chainlink_price: Optional[float] = None

    @property
    def time_remaining(self) -> float:
        """Get seconds remaining until settlement (negative = seconds past expiry)."""
        now = datetime.now(timezone.utc)
        return (self.end_time - now).total_seconds()

    @property
    def is_active(self) -> bool:
        """Check if market is still active for trading."""
        return self.time_remaining > 0

    @property
    def implied_prob_up(self) -> float:
        """Get implied probability of UP from market odds."""
        # Cost to buy UP = best_ask
        return self.best_ask

    @property
    def implied_prob_down(self) -> float:
        """Get implied probability of DOWN from market odds."""
        # Cost to buy DOWN = 1 - best_bid for UP
        return 1 - self.best_bid


@dataclass
class Signal:
    """Trading signal generated by the strategy engine."""

    # Market reference
    market: MarketState

    # Signal details
    side: Side                  # UP, DOWN, or NONE
    edge: float                 # Edge over market odds (0.0 - 1.0)
    true_prob: float            # Calculated true probability
    market_prob: float          # Current market probability

    # Execution recommendation
    recommended_action: OrderAction
    recommended_price: float    # Price to post order at
    size_usd: float             # Position size in USD
    size_shares: float          # Position size in shares

    # Context
    chainlink_price: float      # Current Chainlink price
    time_remaining: float       # Seconds until settlement
    reasoning: str              # Human-readable explanation

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    signal_id: str = field(default_factory=lambda: "")

    def __post_init__(self):
        """Generate signal ID if not provided."""
        if not self.signal_id:
            ts = self.timestamp.strftime("%Y%m%d%H%M%S%f")
            self.signal_id = f"sig_{self.market.asset}_{self.side.value}_{ts}"


@dataclass
class Position:
    """Open position in a market."""

    market: MarketState
    side: Side                  # UP or DOWN
    token_id: str               # Token ID held
    entry_price: float          # Average entry price
    shares: float               # Number of shares held
    entry_time: datetime        # When position was opened
    unrealized_pnl: float = 0.0 # Current unrealized P&L

    # ML data (stored at entry for outcome recording)
    ml_volatility: Optional[float] = None
    ml_momentum: Optional[float] = None
    ml_confidence: Optional[float] = None
    ml_arb_type: Optional[str] = None
    ml_spread: Optional[float] = None
    ml_bid_depth: Optional[float] = None
    ml_ask_depth: Optional[float] = None
    ml_price_trend: Optional[float] = None
    ml_distance_from_target: Optional[float] = None
    ml_binance_lead_pct: Optional[float] = None
    ml_binance_confirmation: Optional[str] = None
    ml_trend_1h: Optional[float] = None
    ml_trend_4h: Optional[float] = None
    ml_trend_1d: Optional[float] = None

    # Chart analysis data (from Binance candlestick analysis at entry)
    ml_chart_rsi: Optional[float] = None
    ml_chart_trend_strength: Optional[float] = None
    ml_chart_is_uptrend: Optional[float] = None
    ml_chart_is_downtrend: Optional[float] = None
    ml_chart_is_ranging: Optional[float] = None
    ml_chart_bullish_reversal: Optional[float] = None
    ml_chart_bearish_reversal: Optional[float] = None
    ml_chart_momentum: Optional[float] = None
    ml_chart_bias_bullish: Optional[float] = None
    ml_chart_bias_bearish: Optional[float] = None
    ml_chart_confidence: Optional[float] = None
    ml_chart_bullish_pattern: Optional[float] = None
    ml_chart_bearish_pattern: Optional[float] = None

    # NEW INDICATOR FEATURES (MACD, BB, Stochastic, RSI Div, Volume, HA, VWAP)
    ml_macd_histogram: Optional[float] = None
    ml_macd_crossover: Optional[str] = None
    ml_bb_bandwidth: Optional[float] = None
    ml_bb_position: Optional[str] = None
    ml_stoch_k: Optional[float] = None
    ml_stoch_d: Optional[float] = None
    ml_stoch_signal: Optional[str] = None
    ml_rsi_divergence: Optional[str] = None
    ml_rsi_divergence_strength: Optional[float] = None
    ml_volume_ratio: Optional[float] = None
    ml_is_high_volume: Optional[bool] = None
    ml_obv_trend: Optional[float] = None
    ml_ha_trend: Optional[str] = None
    ml_ha_consecutive: Optional[int] = None
    ml_ha_strength: Optional[float] = None
    ml_vwap_distance_pct: Optional[float] = None
    ml_vwap_position: Optional[str] = None

    # Explicit token mapping (set at open time, immutable for settlement)
    market_id: str = ""          # condition_id of market this position was opened in
    yes_token_id: str = ""       # UP token id at open time
    no_token_id: str = ""        # DOWN token id at open time
    held_token_id: str = ""      # the token actually purchased

    # Averaging down tracking
    times_averaged: int = 0  # How many times we've added to this position
    original_entry_price: Optional[float] = None  # First entry price before averaging
    total_cost: float = 0.0  # Total USD spent on position
    last_average_time: Optional[datetime] = None  # When we last averaged down

    @property
    def cost_basis(self) -> float:
        """Get total cost basis."""
        return self.entry_price * self.shares

    @property
    def current_value(self) -> float:
        """Get current position value (at best bid)."""
        if self.side == Side.UP:
            return self.market.best_bid * self.shares
        else:
            return (1 - self.market.best_ask) * self.shares

    def update_unrealized_pnl(self):
        """Update unrealized P&L based on current market."""
        self.unrealized_pnl = self.current_value - self.cost_basis


@dataclass
class Order:
    """Order placed on the CLOB."""

    order_id: str
    token_id: str
    side: str                   # "BUY" or "SELL"
    price: float
    size: float
    order_type: str             # "GTC", "FOK", "FAK"
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_complete(self) -> bool:
        """Check if order is in a terminal state."""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
            OrderStatus.REJECTED,
        ]

    @property
    def fill_pct(self) -> float:
        """Get fill percentage."""
        if self.size == 0:
            return 0.0
        return self.filled_size / self.size


@dataclass
class TradeResult:
    """Result of executing a trade."""

    success: bool
    order_id: Optional[str] = None
    filled_size: float = 0.0
    filled_price: float = 0.0
    fee_paid: float = 0.0
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PriceHistory:
    """Historical price data for volatility calculation."""

    asset: str
    prices: list[tuple[datetime, float]] = field(default_factory=list)

    def add_price(self, timestamp: datetime, price: float):
        """Add a price point."""
        self.prices.append((timestamp, price))
        # Keep only last 100 data points
        if len(self.prices) > 100:
            self.prices = self.prices[-100:]

    def get_prices(self, window: int = 20) -> list[float]:
        """Get recent prices for volatility calculation."""
        return [p[1] for p in self.prices[-window:]]


@dataclass
class DailyStats:
    """Daily trading statistics."""

    date: str  # YYYY-MM-DD
    starting_bankroll: float
    current_bankroll: float
    trades_count: int = 0
    wins: int = 0
    losses: int = 0
    total_volume: float = 0.0
    total_pnl: float = 0.0
    fees_paid: float = 0.0
    rebates_earned: float = 0.0

    @property
    def win_rate(self) -> float:
        """Get win rate."""
        if self.trades_count == 0:
            return 0.0
        return self.wins / self.trades_count

    @property
    def net_pnl(self) -> float:
        """Get net P&L after fees."""
        return self.total_pnl - self.fees_paid + self.rebates_earned

    @property
    def daily_return(self) -> float:
        """Get daily return percentage."""
        if self.starting_bankroll == 0:
            return 0.0
        return self.net_pnl / self.starting_bankroll

    def hit_loss_limit_at(self, threshold: float = 0.20) -> bool:
        """
        Check if daily loss limit has been hit.

        Args:
            threshold: Loss limit as a fraction (e.g. 0.25 = 25%).
        """
        if self.starting_bankroll == 0:
            return False
        return self.daily_return < -threshold

    @property
    def hit_loss_limit(self) -> bool:
        """Check if daily loss limit has been hit (default 20% fallback)."""
        return self.hit_loss_limit_at(0.20)
