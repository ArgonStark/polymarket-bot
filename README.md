# Polymarket 15-Minute Crypto Arbitrage Bot

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

## Configuration

### Required Settings

| Variable | Description |
|----------|-------------|
| `PK` | Private key for your Polymarket wallet |
| `FUNDER` | Your Polymarket deposit address |

### Trading Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `MIN_EDGE` | 0.03 | Minimum edge to trade (3%) |
| `MIN_TIME_REMAINING` | 30 | Min seconds before expiry |
| `BASE_POSITION_SIZE` | 25 | Position size in USD |
| `MAX_POSITION_PCT` | 0.15 | Max position % of bankroll |
| `MAX_CONCURRENT_POSITIONS` | 2 | Max open positions |
| `DAILY_LOSS_LIMIT` | 0.25 | Daily loss limit (25%) |

### Loss Protection Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONSECUTIVE_LOSSES` | 3 | Stop after N consecutive losses |
| `MAX_DRAWDOWN_PCT` | 0.15 | Stop if account drops 15% from peak |
| `MIN_WIN_RATE` | 0.40 | Pause if win rate drops below 40% |
| `MIN_TRADES_FOR_WINRATE` | 5 | Trades before win rate check kicks in |
| `COOLOFF_PERIOD_MINUTES` | 30 | Pause duration after hitting limits |

### Machine Learning Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ML_ENABLED` | true | Enable ML signal filtering |
| `ML_MIN_CONFIDENCE` | 0.55 | Min predicted win probability (55%) |
| `ML_MIN_SAMPLES` | 10 | Training samples before ML kicks in |

### Arbitrage Strategy Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ARB_DUMP_THRESHOLD` | 0.15 | Price drop % that triggers entry (15%) |
| `ARB_HEDGE_THRESHOLD` | 0.95 | Hedge when leg1 + opposite <= 0.95 |
| `ARB_ENTRY_WINDOW` | 2.0 | Focus on first N minutes of period |
| `ARB_MIN_SPREAD` | 0.025 | Minimum profit after fees (2.5%) |

### Edge and Time Thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `EDGE_FOR_POST_ONLY` | 0.03 | Min edge for maker orders |
| `EDGE_FOR_LIMIT` | 0.08 | Min edge for limit orders |
| `EDGE_FOR_MARKET` | 0.15 | Min edge for market orders |
| `TIME_FOR_MARKET` | 60 | Seconds remaining for market orders |
| `TIME_FOR_LIMIT` | 120 | Seconds remaining for limit orders |

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
