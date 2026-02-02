# Polymarket 15-Minute Crypto Arbitrage Bot

An autonomous trading bot for Polymarket's 15-minute cryptocurrency prediction markets. Exploits temporal arbitrage between real-time Chainlink oracle prices and lagging market odds, enhanced with an 82-feature ML system.

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/your-repo/polymarket-bot.git
cd polymarket-bot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your wallet credentials

# 3. Train ML model (optional but recommended)
python -m src.tools.enrich_historical_trades  # Fetch historical data
python -m src.tools.train_from_enriched       # Train model

# 4. Run
python -m src.main --dry-run  # Test first
python -m src.main            # Live trading
```

## Strategy Overview

### The Edge

15-minute crypto markets on Polymarket settle based on **Chainlink oracle prices** at expiry:

1. **Chainlink prices move** in real-time with the spot market
2. **Polymarket odds lag** because market participants are slow to update
3. **Exploit the gap** between true probability and stale odds

### Settlement Rules

- **UP wins**: End price >= Start price (per Chainlink)
- **DOWN wins**: End price < Start price (per Chainlink)

### Arbitrage Strategies

Research on top Polymarket traders revealed key patterns:
- Bots earning **$5-10k daily** on 15-minute crypto markets
- **98% win rate** achieved by exploiting price lag

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
│  Chainlink RTDS        │  Polymarket CLOB WS  │  Binance   │
│  (Settlement Price)    │  (Order Book)        │  (Charts)  │
│  wss://ws-live-data    │  wss://ws-clob       │  REST API  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     ML SYSTEM (82 Features)                 │
├─────────────────────────────────────────────────────────────┤
│  Gradient Boosting     │  Neural Network      │  Ensemble  │
│  - L2 regularization   │  - LSTM-style memory │  - Hybrid  │
│  - Feature importance  │  - Sequence patterns │  - Voting  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     STRATEGY ENGINE                         │
├─────────────────────────────────────────────────────────────┤
│  Signal Generator      │  Arbitrage Detector  │ Risk Mgmt  │
│  - Probability calc    │  - Binary mispricing │ - Kelly    │
│  - Edge detection      │  - Dump detection    │ - Drawdown │
│  - Multi-timeframe     │  - Hedge execution   │ - Cooloff  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     EXECUTION LAYER                         │
├─────────────────────────────────────────────────────────────┤
│  OrderExecutor         │  py-clob-client                   │
│  - POST_ONLY (maker)   │  - L2 authenticated               │
│  - GTC limit orders    │  - Polygon network                │
│  - FOK market orders   │                                   │
└─────────────────────────────────────────────────────────────┘
```

## Machine Learning System

The bot includes a sophisticated hybrid ML system with 82 features and ensemble prediction.

### Feature Categories (82 Total)

| Category | Features | Description |
|----------|----------|-------------|
| **Core** | 8 | Edge, time remaining, volatility, momentum, hour, day, asset, side |
| **Market Microstructure** | 5 | Spread, bid depth, ask depth, arb type, distance from target |
| **Multi-Timeframe Trends** | 6 | Binance lead %, confirmation, 1h/4h/1d trends |
| **Chart Analysis** | 12 | RSI, trend strength, direction flags, reversal signals, momentum, bias, patterns |
| **Technical Indicators** | 17 | MACD, Bollinger Bands, Stochastic, RSI divergence, volume, Heiken Ashi, VWAP |
| **Price Context** | 5 | Current price, target price, high, low, velocity |
| **One-Hot Encoding** | ~29 | Asset types, arb types, categorical indicators |

### Hybrid Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID ENSEMBLE                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────┐    ┌─────────────────────┐        │
│  │  Gradient Boosting  │    │   Neural Network    │        │
│  │  ─────────────────  │    │  ─────────────────  │        │
│  │  • 100 estimators   │    │  • 3 hidden layers  │        │
│  │  • Max depth: 4     │    │  • LSTM-style mem   │        │
│  │  • L2 regularization│    │  • Dropout: 0.3     │        │
│  │  • Learning: 0.1    │    │  • Sequence: 10     │        │
│  └──────────┬──────────┘    └──────────┬──────────┘        │
│             │                          │                    │
│             └──────────┬───────────────┘                    │
│                        ▼                                    │
│              ┌─────────────────┐                            │
│              │ Weighted Average│                            │
│              │   GB: 60%       │                            │
│              │   NN: 40%       │                            │
│              └────────┬────────┘                            │
│                       ▼                                     │
│              Final Prediction                               │
└─────────────────────────────────────────────────────────────┘
```

### Clean ML Interface

The ML system uses a clean interface layer for separation of concerns:

```python
from src.ml import MLPredictor, create_ml_input_from_signal, NoOpML

# Production usage
ml = MLPredictor(min_confidence=0.52, min_samples=30)
ml_input = create_ml_input_from_signal(signal, market, chart_analysis, signal_generator)
decision = ml.evaluate_trade(ml_input)

if decision.should_trade:
    execute_trade()

# Record outcome for learning
ml.record_outcome(trade_id, ml_input, won=True)

# For testing/disabled mode
ml = NoOpML()  # Approves all trades
```

### Training Pipeline

Train the ML model from historical data of successful traders:

```bash
# Step 1: Fetch and enrich historical trades
# This fetches trades from a successful trader and adds Binance market context
python -m src.tools.enrich_historical_trades

# Step 2: Train the model
python -m src.tools.train_from_enriched
```

The enrichment process adds:
- Binance OHLCV data at trade time
- Technical indicators (RSI, MACD, BB, etc.)
- Market conditions and trends

### ML Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ML_ENABLED` | true | Enable ML signal filtering |
| `ML_MIN_CONFIDENCE` | 0.55 | Min predicted win probability |
| `ML_MIN_SAMPLES` | 10 | Training samples before ML activates |

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

### Wallet Configuration (Required)

| Variable | Description |
|----------|-------------|
| `PK` | Private key for your wallet (with 0x prefix) |
| `FUNDER` | Polymarket deposit address (only for proxy wallets) |

**Wallet Types:**
- **EOA Wallet**: Standard Ethereum wallet. Leave `FUNDER` empty.
- **Proxy Wallet**: Polymarket browser wallet. Set `FUNDER` to your deposit address.

### Trading Modes

| Variable | Default | Description |
|----------|---------|-------------|
| `TRADING_MODE` | `normal` | Mode: `conservative`, `normal`, or `aggressive` |

| Setting | Conservative | Normal | Aggressive |
|---------|-------------|--------|------------|
| `MIN_EDGE` | 0.08 | 0.03 | 0.02 |
| `BASE_POSITION_SIZE` | 15 | 25 | 50 |
| `MAX_POSITION_PCT` | 0.08 | 0.15 | 0.25 |
| `MAX_CONCURRENT_POSITIONS` | 4 | 8 | 12 |

### Core Trading Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `MIN_EDGE` | 0.03 | Minimum edge to trade (3%) |
| `MIN_TIME_REMAINING` | 60 | Min seconds before expiry |
| `BASE_POSITION_SIZE` | 10 | Base position size in USD |
| `MAX_POSITION_PCT` | 0.10 | Max position as % of bankroll |
| `MAX_CONCURRENT_POSITIONS` | 8 | Max open positions |
| `DAILY_LOSS_LIMIT` | 0.25 | Daily loss limit (25%) |

### Edge and Order Types

| Variable | Default | Description |
|----------|---------|-------------|
| `EDGE_FOR_POST_ONLY` | 0.03 | Min edge for maker orders (rebates) |
| `EDGE_FOR_LIMIT` | 0.08 | Min edge for limit orders |
| `EDGE_FOR_MARKET` | 0.15 | Min edge for market orders |
| `TIME_FOR_MARKET` | 60 | Seconds remaining for market orders |

### Loss Protection

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONSECUTIVE_LOSSES` | 3 | Stop after N consecutive losses |
| `MAX_DRAWDOWN_PCT` | 0.15 | Stop if account drops 15% from peak |
| `MIN_WIN_RATE` | 0.40 | Pause if win rate below 40% |
| `COOLOFF_PERIOD_MINUTES` | 30 | Pause duration after hitting limits |

### Trend Protection

| Variable | Default | Description |
|----------|---------|-------------|
| `TREND_PROTECTION_ENABLED` | true | Enable trend protection |
| `MAX_OPPOSITE_TREND_1D` | 0.25 | Block if 1d trend > 25% against signal |
| `MAX_OPPOSITE_TREND_1H` | 0.40 | Block if 1h trend > 40% against signal |
| `VELOCITY_GUARD_ENABLED` | true | Enable velocity guard |
| `BINANCE_MOMENTUM_ENABLED` | true | Enable Binance momentum signals |

### Early Exit (Experimental)

| Variable | Default | Description |
|----------|---------|-------------|
| `EARLY_EXIT_ENABLED` | false | Enable early exit |
| `TAKE_PROFIT_PCT` | 0.30 | Take profit at +30% |
| `STOP_LOSS_PCT` | 0.25 | Stop loss at -25% |

### Bot Mode

| Variable | Default | Description |
|----------|---------|-------------|
| `DRY_RUN` | true | Run without placing actual trades |

## Risk Management

### Built-in Protections

| Protection | Default | Trigger |
|------------|---------|---------|
| **Daily Loss Limit** | 25% | Halts trading for the day |
| **Consecutive Losses** | 3 | Enters 30-minute cooloff |
| **Max Drawdown** | 15% | Enters cooloff |
| **Win Rate Check** | 40% | Pauses if win rate too low |
| **Kelly Sizing** | Enabled | Optimal position sizing based on ML confidence |

### Connection Resilience

- **Circuit Breaker**: WebSocket feeds auto-reconnect with exponential backoff
- **Max Retries**: 10 consecutive failures opens circuit breaker
- **Feed Monitoring**: Trading halts if both Chainlink and CLOB feeds fail

## Usage

### Commands

```bash
# Simulation mode (recommended first)
python -m src.main --dry-run

# Live trading
python -m src.main

# With options
python -m src.main --log-level DEBUG --min-edge 0.05
```

### Training Tools

```bash
# Enrich historical trades with market data
python -m src.tools.enrich_historical_trades

# Train ML model from enriched data
python -m src.tools.train_from_enriched
```

### Analysis Tools

```bash
# Analyze a trader's wallet
python analyze_wallet.py 0x1234...abcd
python analyze_wallet.py 0x1234...abcd --detailed
```

### Copy Trading

```bash
# Copy trades from successful traders
python copy_bot.py -w 0x63ce342161250d705dc0b16df89036c8e5f9ba9a

# With sizing options
python copy_bot.py -w 0x1234... --size-mode kelly --dry-run
```

## Project Structure

```
polymarket-bot/
├── main.py                     # Main entry point
├── copy_bot.py                 # Copy trading bot
├── analyze_wallet.py           # Wallet analysis
├── requirements.txt
├── .env.example
│
├── models/
│   └── ml_model.json           # Trained ML model
│
├── data/
│   └── enriched_trades.json    # Historical trade data
│
└── src/
    ├── config.py               # Configuration
    ├── models.py               # Data models
    ├── bot.py                  # Main trading bot
    │
    ├── ml/                     # ML Interface Layer
    │   ├── __init__.py
    │   ├── interface.py        # Abstract interface (MLInput, MLDecision)
    │   └── predictor.py        # Production implementation
    │
    ├── data/
    │   ├── chainlink.py        # Chainlink price feed
    │   ├── clob.py             # Order book feed
    │   ├── binance.py          # Binance prices
    │   ├── binance_chart.py    # Chart analysis
    │   └── gamma.py            # Market discovery
    │
    ├── execution/
    │   ├── client.py           # Trading client
    │   └── orders.py           # Order execution
    │
    ├── strategy/
    │   ├── signals.py          # Signal generation
    │   ├── risk.py             # Risk management
    │   ├── arbitrage.py        # Arbitrage detection
    │   ├── ml_predictor.py     # ML prediction engine
    │   └── trade_history.py    # Performance tracking
    │
    ├── tools/
    │   ├── enrich_historical_trades.py  # Data enrichment
    │   └── train_from_enriched.py       # Model training
    │
    └── utils/
        └── logging.py          # Logging utilities
```

## Supported Assets

| Asset | Typical 15-min Volatility |
|-------|---------------------------|
| BTC | 0.25% - 0.40% |
| ETH | 0.35% - 0.50% |
| SOL | 0.50% - 0.80% |
| XRP | 0.40% - 0.70% |

## Key Principles

1. **Chainlink is truth** - Settlement uses Chainlink, trade based on Chainlink
2. **Don't predict** - Exploit mispricing, not direction
3. **Speed wins** - Sub-second reaction to price movements
4. **Maker first** - Earn rebates when possible
5. **Kelly sizing** - Optimal position sizing from ML confidence
6. **Survive to trade** - Risk management prevents ruin
7. **Learn from data** - ML improves with every outcome

## Technical Analysis Features

- **Trend Detection**: EMA crossovers, price momentum
- **RSI Extremes**: Overbought (>70) / Oversold (<30)
- **Multi-Timeframe**: 15m, 1h, 4h trend alignment
- **MACD**: Histogram and crossover signals
- **Bollinger Bands**: Bandwidth and position
- **Stochastic**: K/D crossovers
- **RSI Divergence**: Bullish/bearish divergence detection
- **Volume Analysis**: Ratio, OBV trend
- **Heiken Ashi**: Trend confirmation
- **VWAP**: Distance and position

## Disclaimer

This software is for educational purposes only. Trading cryptocurrency derivatives involves substantial risk of loss. Only trade with funds you can afford to lose. The authors are not responsible for any financial losses incurred through use of this software.

## License

MIT License
