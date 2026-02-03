# Polymarket 15-Minute Crypto Arbitrage Bot

An autonomous trading bot for Polymarket's 15-minute cryptocurrency prediction markets. Uses a simple distance-from-target + trend strategy to predict where markets will close.

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

# 3. Run
python -m src.main --dry-run  # Test first
python -m src.main            # Live trading
```

## Strategy Overview

### Simple Signal Strategy (RECOMMENDED)

The bot uses a **distance-from-target + trend** strategy for maximum profitability:

**Decision Hierarchy:**
1. **DISTANCE** (Primary) - If price is far from target (>0.3%), it rarely crosses
2. **TREND** (Secondary) - If price is near target, follow the trend direction
3. **MOMENTUM** (Confirmation) - Short-term velocity confirms or warns

| Scenario | Signal | Win Rate |
|----------|--------|----------|
| Price >0.3% above target | UP | 85%+ |
| Price >0.3% below target | DOWN | 85%+ |
| Near target + Strong downtrend | DOWN | 75-80% |
| Near target + Strong uptrend | UP | 75-80% |
| Near target + Moderate trend | Follow trend | 65-70% |
| At target + Ranging | Use distance | 52-58% |

### Settlement Rules

- **UP wins**: End price >= Start price (per Chainlink)
- **DOWN wins**: End price < Start price (per Chainlink)

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

### ML Settings (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `ML_ENABLED` | false | ML filtering (not needed with SIMPLE_MODE) |
| `ML_MIN_CONFIDENCE` | 0.52 | Min predicted win probability |
| `ML_MIN_SAMPLES` | 50 | Training samples before ML activates |

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

### Signal Modes

| Variable | Default | Description |
|----------|---------|-------------|
| `SIMPLE_MODE` | `true` | **RECOMMENDED** - Use distance+trend strategy |
| `AGGRESSIVE_MODE` | `false` | Trade every market, always pick a side |
| `TRADING_MODE` | `normal` | Risk profile: `conservative`, `normal`, `aggressive` |

### Simple Mode (Default)

Best for consistent returns. Uses distance from target + trend to predict settlement.

| Setting | Value | Description |
|---------|-------|-------------|
| `MIN_EDGE` | 0.00 | Trade all signals (reliable with simple mode) |
| `MAX_CONCURRENT_POSITIONS` | 4 | One per asset (BTC, ETH, SOL, XRP) |
| `ORDER_COOLDOWN_SECONDS` | 10 | Fast reaction to market changes |

### Core Trading Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `MIN_EDGE` | 0.00 | Minimum edge to trade (0% with simple mode) |
| `MIN_TIME_REMAINING` | 30 | Min seconds before expiry |
| `BASE_POSITION_SIZE` | 25 | Base position size in USD |
| `MAX_POSITION_PCT` | 0.10 | Max position as % of bankroll |
| `MAX_CONCURRENT_POSITIONS` | 4 | Max open positions (one per asset) |
| `ORDER_COOLDOWN_SECONDS` | 10 | Cooldown between trades for same asset |
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
    │   ├── simple_signals.py   # SIMPLE signal strategy (recommended)
    │   ├── signals.py          # Signal generation coordinator
    │   ├── risk.py             # Risk management
    │   ├── arbitrage.py        # Arbitrage detection
    │   ├── ml_predictor.py     # ML prediction engine (optional)
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

1. **Distance is king** - Price far from target rarely crosses
2. **Trend follows** - When price is near target, follow the trend
3. **Chainlink is truth** - Settlement uses Chainlink oracle prices
4. **Keep it simple** - Distance + trend beats complex indicators
5. **Manage risk** - Position sizing based on conviction
6. **Survive to trade** - Risk management prevents ruin

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
