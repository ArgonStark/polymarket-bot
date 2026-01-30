# Polymarket 15-Minute Crypto Arbitrage Bot

**Made by Argon Stark**

An autonomous trading bot for Polymarket's 15-minute cryptocurrency prediction markets. Exploits temporal arbitrage between real-time Chainlink oracle prices and lagging market odds.

## Strategy Overview

### The Edge

15-minute crypto markets on Polymarket settle based on **Chainlink oracle prices** at expiry. The key insight:

1. **Chainlink prices move** in real-time with the spot market
2. **Polymarket odds lag** because market participants are slow to update
3. **Exploit the gap** between true probability and stale odds

### Settlement Rules

- **UP wins**: End price >= Start price (per Chainlink)
- **DOWN wins**: End price < Start price (per Chainlink)
- Settlement is deterministic once the 15 minutes expire

### NEW: Arbitrage Strategy (Based on Successful Bots)

Research on top Polymarket traders revealed key patterns:
- Bots earning **$5-10k daily** on 15-minute crypto markets
- **98% win rate** achieved by exploiting price lag
- **$313 → $414k** in one month using these strategies

**Key insight: Don't predict direction - exploit mispricing.**

| Strategy | How It Works |
|----------|--------------|
| **Binary Mispricing** | Buy both YES+NO when total < $1.00 for guaranteed profit |
| **Asymmetric Pricing** | Buy whichever side is temporarily too cheap |
| **Dump Detection** | Enter when price drops 15%+ in 3 seconds |
| **Hedge Execution** | Lock in profit when leg1 + opposite <= 0.95 |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       DATA LAYER                            │
├─────────────────────────────────────────────────────────────┤
│  Chainlink RTDS        │  Polymarket CLOB WS  │  Gamma API │
│  (Settlement Price)    │  (Order Book)        │  (Markets) │
│  wss://ws-live-data.   │  wss://ws-clob...    │  REST API  │
│  polymarket.com        │                       │            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     STRATEGY ENGINE                          │
├─────────────────────────────────────────────────────────────┤
│  Signal Generator      │  Arbitrage Detector  │ ML Predictor│
│  - Probability calc    │  - Binary mispricing │ - Win prob  │
│  - Edge detection      │  - Dump detection    │ - Learning  │
│  - Action decision     │  - Hedge execution   │ - Filtering │
├─────────────────────────────────────────────────────────────┤
│                       RISK MANAGER                           │
│  - Position limits     │  - Consecutive loss  │ - Drawdown  │
│  - Daily loss limit    │  - Win rate check    │ - Cooloff   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     EXECUTION LAYER                          │
├─────────────────────────────────────────────────────────────┤
│  OrderExecutor         │  py-clob-client                    │
│  - POST_ONLY (maker)   │  - L2 authenticated                │
│  - GTC limit           │  - Polygon network                 │
│  - FOK market          │                                    │
└─────────────────────────────────────────────────────────────┘
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repo/polymarket-bot.git
cd polymarket-bot
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your wallet credentials
```

## Environment Configuration (.env)

The bot uses a `.env` file for configuration. Copy `.env.example` to `.env` and customize for your setup:

```bash
cp .env.example .env
```

### Wallet Configuration (Required)

| Variable | Description |
|----------|-------------|
| `PK` | Private key for your wallet (with 0x prefix) |
| `FUNDER` | Polymarket deposit address (only for proxy wallets) |

**Wallet Types:**
- **EOA Wallet**: Standard Ethereum wallet (MetaMask). Leave `FUNDER` empty. Uses `signature_type=0`
- **Proxy Wallet**: Polymarket browser/Magic wallet. Set `FUNDER` to your deposit address. Uses `signature_type=1`

Find your deposit address at: https://polymarket.com/wallet

### API Credentials (Optional)

If you've already generated API credentials, set them to avoid re-generating on startup:

| Variable | Description |
|----------|-------------|
| `CLOB_API_KEY` | Polymarket CLOB API key |
| `CLOB_SECRET` | Polymarket CLOB API secret |
| `CLOB_PASS_PHRASE` | Polymarket CLOB API passphrase |

### Trading Mode

| Variable | Default | Description |
|----------|---------|-------------|
| `TRADING_MODE` | `normal` | Mode: `conservative`, `normal`, or `aggressive` |

**Mode Comparison:**

| Setting | Conservative | Normal | Aggressive |
|---------|-------------|--------|------------|
| `MIN_EDGE` | 0.08 | 0.03 | 0.02 |
| `BASE_POSITION_SIZE` | 15 | 25 | 50 |
| `MAX_POSITION_PCT` | 0.08 | 0.15 | 0.25 |
| `MAX_CONCURRENT_POSITIONS` | 4 | 8 | 12 |

> **Warning**: Aggressive mode is HIGH RISK. You can lose your entire bankroll.

### Trading Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `MIN_EDGE` | 0.03 | Minimum edge to trade (3%) |
| `MIN_TIME_REMAINING` | 60 | Min seconds before expiry |
| `BASE_POSITION_SIZE` | 10 | Base position size in USD |
| `MAX_POSITION_PCT` | 0.10 | Max position as % of bankroll |
| `MAX_CONCURRENT_POSITIONS` | 8 | Max open positions at once |
| `DAILY_LOSS_LIMIT` | 0.25 | Daily loss limit (25% of bankroll) |
| `ORDER_COOLDOWN_SECONDS` | 5 | Cooldown between trades for same asset |
| `ASSET_PRIORITY` | `BTC,ETH,SOL,XRP` | Priority order for asset checking |
| `PARALLEL_EXECUTION` | true | Enable parallel market processing |
| `LOOP_INTERVAL` | 0.25 | Trading loop interval in seconds |

### Edge and Time Thresholds

**Important**: Polymarket has dynamic taker fees up to 3.15% near 50% odds. POST_ONLY orders earn maker rebates instead of paying fees.

| Variable | Default | Description |
|----------|---------|-------------|
| `EDGE_FOR_POST_ONLY` | 0.03 | Min edge for maker orders (rebates) |
| `EDGE_FOR_LIMIT` | 0.08 | Min edge for limit orders |
| `EDGE_FOR_MARKET` | 0.15 | Min edge for market orders (pays fees) |
| `TIME_FOR_MARKET` | 60 | Seconds remaining threshold for market orders |
| `TIME_FOR_LIMIT` | 120 | Seconds remaining threshold for limit orders |
| `MAKER_ORDER_TIMEOUT` | 30 | Maker order timeout before cancellation |

### Loss Protection Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONSECUTIVE_LOSSES` | 3 | Stop after N consecutive losses |
| `MAX_DRAWDOWN_PCT` | 0.15 | Stop if account drops 15% from peak |
| `MIN_WIN_RATE` | 0.40 | Pause if win rate drops below 40% |
| `MIN_TRADES_FOR_WINRATE` | 5 | Trades required before win rate check |
| `COOLOFF_PERIOD_MINUTES` | 30 | Pause duration after hitting limits |

### Machine Learning Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ML_ENABLED` | true | Enable ML signal filtering |
| `ML_MIN_CONFIDENCE` | 0.55 | Min predicted win probability (55%) |
| `ML_MIN_SAMPLES` | 10 | Training samples before ML kicks in |

### Trade History Review Settings

The bot reviews past performance before taking new trades:

| Variable | Default | Description |
|----------|---------|-------------|
| `HISTORY_MIN_TRADES` | 5 | Min trades before history filter kicks in |
| `HISTORY_MIN_WIN_RATE` | 0.35 | Min win rate for asset/side to continue |
| `HISTORY_MIN_PREDICTION_ACCURACY` | 0.45 | Min prediction accuracy required |

### Early Exit Settings (Experimental)

Close positions early to lock in profits or limit losses:

| Variable | Default | Description |
|----------|---------|-------------|
| `EARLY_EXIT_ENABLED` | false | Enable early exit feature |
| `TAKE_PROFIT_PCT` | 0.30 | Take profit at +30% gain |
| `STOP_LOSS_PCT` | 0.25 | Stop loss at -25% loss |
| `EARLY_EXIT_MIN_HOLD` | 60 | Min seconds before allowing early exit |
| `EARLY_EXIT_INTERVAL` | 5 | Check interval for early exit (seconds) |

### Arbitrage Strategy Settings

Based on successful bot patterns that made $5-10k daily:

| Variable | Default | Description |
|----------|---------|-------------|
| `ARB_DUMP_THRESHOLD` | 0.15 | Price drop % that triggers entry (15%) |
| `ARB_HEDGE_THRESHOLD` | 0.95 | Hedge when leg1 + opposite <= 0.95 |
| `ARB_ENTRY_WINDOW` | 2.0 | Focus on first N minutes of period |
| `ARB_MIN_SPREAD` | 0.025 | Minimum profit after fees (2.5%) |

### Observation/Warm-up Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `WARMUP_PERIOD` | 60 | Warm-up period at startup (seconds) |
| `MIN_PRICE_SAMPLES` | 10 | Min price samples needed before trading |
| `MIN_OBSERVATION_TIME` | 30 | Min observation time per market (seconds) |

### Trend Protection Settings

Avoid trading against strong market trends:

| Variable | Default | Description |
|----------|---------|-------------|
| `TREND_PROTECTION_ENABLED` | true | Master enable/disable switch |
| `MAX_OPPOSITE_TREND_1D` | 0.25 | Block if 1d trend > 25% against signal |
| `MAX_OPPOSITE_TREND_1H` | 0.40 | Block if 1h trend > 40% against signal |

**Price Velocity Guard:**

| Variable | Default | Description |
|----------|---------|-------------|
| `VELOCITY_GUARD_ENABLED` | true | Enable velocity guard |
| `VELOCITY_MAX_BTC` | 0.00017 | Max BTC velocity (~1%/min) |
| `VELOCITY_MAX_ETH` | 0.00025 | Max ETH velocity (~1.5%/min) |
| `VELOCITY_MAX_SOL` | 0.00033 | Max SOL velocity (~2%/min) |
| `VELOCITY_MAX_XRP` | 0.00030 | Max XRP velocity (~1.8%/min) |

**Multi-Timeframe Agreement:**

| Variable | Default | Description |
|----------|---------|-------------|
| `TIMEFRAME_AGREEMENT_ENABLED` | true | Enable timeframe agreement |
| `EDGE_BOOST_ALL_ALIGNED` | 0.01 | +1% edge when all timeframes agree |
| `EDGE_PENALTY_MIXED` | 0.01 | -1% edge when signals mixed |

**Binance Momentum:**

| Variable | Default | Description |
|----------|---------|-------------|
| `BINANCE_MOMENTUM_ENABLED` | true | Enable Binance momentum |
| `BINANCE_VELOCITY_THRESHOLD` | 0.001 | Velocity threshold (0.1%/sec) |
| `BINANCE_MOMENTUM_BOOST` | 0.02 | +2% edge boost for strong momentum |
| `BINANCE_DIRECT` | false | Use direct Binance WebSocket |

**Dynamic Edge Requirement:**

| Variable | Default | Description |
|----------|---------|-------------|
| `DYNAMIC_EDGE_ENABLED` | true | Enable dynamic edge |
| `DYNAMIC_EDGE_MIN` | 0.015 | Never go below 1.5% |
| `DYNAMIC_EDGE_MAX` | 0.05 | Never exceed 5% |

**Consecutive Move Detection:**

| Variable | Default | Description |
|----------|---------|-------------|
| `CONSECUTIVE_ENABLED` | true | Enable consecutive detection |
| `CONSECUTIVE_MIN_MOVES` | 3 | Min moves to trigger boost |
| `CONSECUTIVE_BOOST_MAX` | 0.02 | Max +2% edge boost |

### WebSocket Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `WS_MAX_RETRIES` | 10 | Max reconnection attempts |
| `WS_RECONNECT_DELAY` | 1.0 | Initial reconnect delay (seconds) |
| `WS_MAX_RECONNECT_DELAY` | 60.0 | Max reconnect delay (seconds) |
| `WS_PING_INTERVAL` | 30 | Ping interval for keepalive (seconds) |
| `WS_PING_TIMEOUT` | 10 | Ping timeout (seconds) |

### Bot Mode

| Variable | Default | Description |
|----------|---------|-------------|
| `DRY_RUN` | true | Run without placing actual trades |

> **Recommended**: Start with `DRY_RUN=true` to test your configuration!

### Notifications (Optional)

| Variable | Description |
|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Telegram bot token for notifications |
| `TELEGRAM_CHAT_ID` | Telegram chat ID for notifications |
| `DISCORD_WEBHOOK_URL` | Discord webhook URL for notifications |

---

## Configuration Presets

### Conservative Mode (Safer Trading)

```bash
MIN_EDGE=0.08
EDGE_FOR_POST_ONLY=0.08
EDGE_FOR_LIMIT=0.15
EDGE_FOR_MARKET=0.25
BASE_POSITION_SIZE=15
MAX_POSITION_PCT=0.08
MAX_CONCURRENT_POSITIONS=4
```

### Aggressive Mode (Higher Risk)

```bash
MIN_EDGE=0.02
EDGE_FOR_POST_ONLY=0.02
EDGE_FOR_LIMIT=0.05
EDGE_FOR_MARKET=0.10
BASE_POSITION_SIZE=50
MAX_POSITION_PCT=0.25
MAX_CONCURRENT_POSITIONS=12
DAILY_LOSS_LIMIT=0.35
```

> **Warning**: Aggressive mode is HIGH RISK. Only use if you understand the risks!

## Machine Learning System

The bot includes an online learning ML model that improves with each trade.

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                            │
│                                                              │
│  Signal → Extract Features → Record with Outcome             │
│                                                              │
│  Features:                                                   │
│  ├── edge (expected profit)        → normalized 0-1          │
│  ├── time_remaining (seconds)      → normalized 0-1          │
│  ├── volatility                    → normalized 0-1          │
│  ├── price_momentum (-1 to +1)     → normalized 0-1          │
│  ├── hour_of_day (0-23)           → normalized 0-1          │
│  ├── day_of_week (0-6)            → normalized 0-1          │
│  ├── asset (BTC/ETH/SOL/XRP)      → one-hot encoded         │
│  └── side (UP/DOWN)               → binary                   │
│                                                              │
│  Outcome: WIN (1) or LOSS (0)                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    PREDICTION PHASE                          │
│                                                              │
│  New Signal → Extract Features → Predict Win Probability     │
│                                                              │
│  Model: Logistic Regression (online learning)                │
│                                                              │
│  prediction = sigmoid(bias + Σ(weight_i × feature_i))        │
│                                                              │
│  If prediction >= 55% → ALLOW trade                          │
│  If prediction < 55%  → REJECT trade                         │
└─────────────────────────────────────────────────────────────┘
```

### The Algorithm

**Logistic Regression with Online Learning:**

```python
# For each feature vector X and outcome y (1=win, 0=loss):

# 1. Make prediction
z = bias + sum(weight[i] * X[i] for i in range(n_features))
prediction = 1 / (1 + exp(-z))  # sigmoid

# 2. Calculate error
error = y - prediction

# 3. Update weights (gradient descent)
for i in range(n_features):
    weight[i] += learning_rate * error * X[i]
bias += learning_rate * error
```

### Learning Phases

| Phase | Trades | Behavior |
|-------|--------|----------|
| **Learning** | 0-9 | All trades allowed, collecting data |
| **Active** | 10+ | Only trades with >55% predicted win |

### Example Output

```
🤖 ML Model: 15 samples | Accuracy: 67% (10/15)
[BTC] 🤖 ML rejected: 42% < 55%
[ETH] 🤖 ML confidence: 68%
    ✓ FILLED @ 0.52 (ML: 68%)
```

### Model Persistence

The model saves to `ml_model.json` every 10 trades:
- Weights and bias preserved
- Training samples count
- Accuracy statistics

## Risk Management

### Built-in Protections

| Protection | Default | Trigger |
|------------|---------|---------|
| **Daily Loss Limit** | 25% | Halts trading for the day |
| **Consecutive Losses** | 3 | Enters 30-minute cooloff |
| **Max Drawdown** | 15% | Enters cooloff if account drops from peak |
| **Win Rate Check** | 40% | Pauses if win rate too low (after 5+ trades) |
| **Cooloff Period** | 30 min | Pause after any limit is hit |

### How Cooloff Works

```
Loss #1 → Continue trading
Loss #2 → Continue trading
Loss #3 → 🛑 COOLOFF STARTED: consecutive losses
          Trading paused for 30 minutes until 14:35:00 UTC

... 30 minutes later ...

✅ Cooloff period ended - resetting protection counters
```

### Connection Resilience

- **Circuit Breaker**: WebSocket feeds automatically reconnect with exponential backoff
- **Max Retries**: After 10 consecutive failures, circuit breaker opens
- **Feed Monitoring**: Trading halts if both Chainlink and CLOB feeds fail
- **Balance Sync**: Syncs with exchange every 30 seconds

## Usage

### Run in Simulation Mode (Recommended First)
```bash
python main.py --dry-run
```

### Run Live Trading
```bash
python main.py
```

### Command Line Options
```bash
python main.py --help

Options:
  --dry-run           Run without placing actual trades
  --log-level LEVEL   DEBUG, INFO, WARNING, ERROR (default: INFO)
  --log-file PATH     Path to log file
  --min-edge FLOAT    Override minimum edge
  --position-size USD Override position size
```

## Log Output

The bot uses colorful logging for easy monitoring:

```
💰 $85.55 │ 📊 BTC, ETH │ BTC: $87,000 | ETH: $3,200
🎯 ARB [BTC]: ASYMMETRIC | Edge: 5% | Price below target, NO too cheap
>>> BTC UP ▲ │ Edge: 35% │ $87,000 vs $86,500 │ $12.50
    ✓ FILLED @ 0.52 (ML: 68%)
🎉 POSITION CLOSED: BTC UP | WIN | Shares: 24.04 | P&L: +$11.54
```

## Project Structure

```
polymarket-bot/
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── .env.example           # Environment template
├── ml_model.json          # Persisted ML model
├── trades.jsonl           # Trade log
└── src/
    ├── config.py          # Configuration management
    ├── models.py          # Data models
    ├── probability.py     # Probability calculations
    ├── bot.py             # Main trading bot
    ├── data/
    │   ├── chainlink.py   # Chainlink price feed
    │   ├── clob.py        # Order book feed
    │   └── gamma.py       # Market discovery
    ├── execution/
    │   ├── client.py      # Trading client
    │   └── orders.py      # Order execution
    ├── strategy/
    │   ├── signals.py     # Signal generation
    │   ├── risk.py        # Risk management
    │   ├── arbitrage.py   # Arbitrage detection
    │   └── ml_predictor.py # ML signal filter
    └── utils/
        └── logging.py     # Logging utilities
```

## Supported Assets

- BTC/USD
- ETH/USD
- SOL/USD
- XRP/USD

### Default Volatilities (15-minute)

| Asset | Typical Range | Default |
|-------|---------------|---------|
| BTC | 0.25% - 0.40% | 0.35% |
| ETH | 0.35% - 0.50% | 0.45% |
| SOL | 0.50% - 0.80% | 0.70% |
| XRP | 0.40% - 0.70% | 0.60% |

## Key Principles

- **Chainlink is truth** - Settlement uses Chainlink, trade based on Chainlink
- **Don't predict** - Exploit mispricing, not direction
- **Speed wins** - Sub-second reaction to price movements
- **Maker first** - Earn rebates when possible
- **Mechanical execution** - Same logic every trade
- **Small edges compound** - Many 3% edge trades > few 50% edge trades
- **Survive to trade** - Risk management prevents ruin
- **Learn from losses** - ML improves with every outcome

## Disclaimer

This software is for educational purposes only. Trading cryptocurrency derivatives involves substantial risk of loss. Only trade with funds you can afford to lose. The authors are not responsible for any financial losses incurred through use of this software.

## License

MIT License
