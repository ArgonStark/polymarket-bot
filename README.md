# Polymarket Crypto Arbitrage Bot

An autonomous trading bot for Polymarket's short-duration cryptocurrency prediction markets. Supports both **15-minute** and **5-minute** market variants. Uses distance-from-target + trend strategy with optional ML filtering, real-time WebSocket feeds, and a full risk management stack.

## Quick Start

```bash
# 1. Clone and setup
git clone <repo-url>
cd polymarket-bot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your wallet credentials and preferred trading profile

# 3. Run in paper trading mode (recommended first)
python -m src.main

# 4. Open the monitoring dashboard
# Terminal UI:  python monitoring/terminal_ui.py
# Web UI:       http://localhost:8766
```

## Strategy Overview

The bot trades Polymarket's crypto prediction markets where settlement is based on whether an asset's price goes UP or DOWN relative to its opening price (per Chainlink oracle).

### Signal Strategy

**Decision Hierarchy:**
1. **DISTANCE** (Primary) -- If price is far from the target (>0.3%), it rarely crosses back
2. **TREND** (Secondary) -- If price is near the target, follow the trend direction
3. **MOMENTUM** (Confirmation) -- Short-term velocity confirms or warns

| Scenario | Signal | Approximate Win Rate |
|----------|--------|----------------------|
| Price >0.3% above target | UP | 85%+ |
| Price >0.3% below target | DOWN | 85%+ |
| Near target + strong trend | Follow trend | 75-80% |
| Near target + moderate trend | Follow trend | 65-70% |
| At target + ranging | Use distance | 52-58% |

### Market Variants

| Variant | Duration | Env Setting | Notes |
|---------|----------|-------------|-------|
| **15-minute** | 900s | `TRADING_VARIANTS=fifteen` | Default, more data per market |
| **5-minute** | 300s | `TRADING_VARIANTS=five` | Faster turnover, tighter windows |
| **Both** | Mixed | `TRADING_VARIANTS=fifteen,five` | Trades all available markets |

5-minute markets use separate timing parameters (`MIN_TIME_REMAINING_5M`, `MIN_OBSERVATION_TIME_5M`) since their shorter duration requires faster entry.

### Settlement Rules

- **UP wins**: End price >= Start price (per Chainlink)
- **DOWN wins**: End price < Start price (per Chainlink)

## Architecture

```
                          DATA LAYER
 ┌──────────────────┬──────────────────┬──────────────────┐
 │  Chainlink RTDS  │  Polymarket CLOB │  Binance REST    │
 │  (Oracle Price)  │  (Order Book WS) │  (Chart/Candles) │
 └────────┬─────────┴────────┬─────────┴────────┬─────────┘
          │                  │                   │
          ▼                  ▼                   ▼
 ┌─────────────────────────────────────────────────────────┐
 │              STRATEGY ENGINE + ML (83 Features)         │
 │  Signal Generator │ Arbitrage Detector │ ML Predictor   │
 │  Risk Manager     │ Capital Scaling    │ Kill Switch    │
 └────────────────────────────┬────────────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 │                   EXECUTION LAYER                       │
 │  Paper Executor (simulated)  │  Live Executor (CLOB)   │
 │  POST_ONLY maker orders      │  GTC limit / FOK market │
 └────────────────────────────────────────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 │                   MONITORING                            │
 │  WebSocket Server │ Rich Terminal UI │ Web Dashboard    │
 │  State Persistence │ Metrics Snapshots │ Structured Logs│
 └─────────────────────────────────────────────────────────┘
```

## Risk Management

### Protection Layers

| Protection | Default | Behavior |
|------------|---------|----------|
| **Daily Loss Limit** | 25% | Halts trading for the rest of the day |
| **Consecutive Losses** | 3 | Enters 30-minute cooloff |
| **Max Drawdown** | 15% from peak | Enters cooloff, then gradual peak decay |
| **Win Rate Floor** | 40% (after 5 trades) | Pauses until rate recovers |
| **Kill Switch** | Enabled | Hard stop on catastrophic loss |
| **Data Health Gate** | Always on | Blocks trading on stale/missing feeds |
| **CLOB Warmup** | 5 seconds | No trading after WebSocket reconnect |

### Drawdown Recovery (Peak Decay)

When drawdown exceeds the limit, the bot enters a cooloff period. After cooloff expires, if drawdown is still breached, **gradual peak decay** begins:

- Peak bankroll exponentially decays toward current bankroll
- Default rate: 5% of the gap per hour (`PEAK_DECAY_RATE_PER_HOUR`)
- A 20% drawdown resolves below the 15% limit in ~15 hours
- Decay stops immediately if equity hits a new high
- Set `PEAK_DECAY_RATE_PER_HOUR=0` to disable (requires manual `reset_drawdown()`)

### State Persistence

Bot state (bankroll, peak, positions, daily stats) is saved to `bot_state.json` every 60 seconds and on shutdown. On restart:
- Bankroll and peak are restored (peak = max of saved peak and current bankroll)
- Open positions are re-linked to markets when rediscovered
- Orphaned positions (expired while offline) are force-closed as losses
- Daily stats reset if a new UTC day has started

### Shutdown Safety

Shutdown follows a strict 5-phase sequence:
1. Stop tick loops
2. Cancel all pending orders
3. Drain in-flight operations
4. Save state + verify invariants
5. Disconnect feeds and monitoring

## Machine Learning System

The bot includes an optional hybrid ML system with **83 features** and ensemble prediction.

### Feature Categories (83 Total)

| Category | Count | Examples |
|----------|-------|---------|
| Core | 6 | Edge, time remaining, volatility, momentum |
| Asset one-hot | 4 | BTC, ETH, SOL, XRP |
| Side + Variant | 2 | UP/DOWN, is_five_min |
| Arb type one-hot | 5 | none, binary_arb, asymmetric, dump, hedge |
| Market microstructure | 4 | Spread, bid/ask depth, price trend |
| Binance signals | 5 | Lead %, confirmation strength |
| Multi-timeframe trends | 3 | 1h, 4h, 1d |
| Price context | 5 | Normalized price, range position, velocity |
| Chart analysis | 20 | RSI, trend, reversals, patterns, uncertainty |
| Technical indicators | 29 | MACD, Bollinger, Stochastic, RSI divergence, Volume, Heiken Ashi, VWAP |

### Model Architecture

- **Random Forest** (pre-trained on successful trader data) -- 11 simple features
- **Neural Network** (online learning from your trades) -- full 83 features
- **Hybrid Predictor** -- adaptive weighting that shifts from RF to NN as training samples grow
- **Gradient Boosting** (ensemble) -- L2 regularized, feature importance tracking

Models auto-migrate when feature count changes (e.g., 82 -> 83 for the variant feature).

### ML Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ML_ENABLED` | `false` | Enable ML confidence filtering |
| `ML_MIN_CONFIDENCE` | `0.52` | Min predicted win probability |
| `ML_MIN_SAMPLES` | `50` | Training samples before ML activates |
| `ML_MIN_EDGE` | `0.02` | Minimum edge override for ML |

## Monitoring

The bot includes a real-time monitoring system that starts automatically.

### Components

| Component | Access | Description |
|-----------|--------|-------------|
| **WebSocket Server** | `ws://localhost:8765` | Broadcasts JSON snapshots to all clients |
| **Web Dashboard** | `http://localhost:8766` | Single-file browser dashboard (dark theme) |
| **Terminal UI** | `python monitoring/terminal_ui.py` | Rich-based terminal dashboard |
| **Mock Data** | `python monitoring/mock_data.py` | Generates fake data for testing dashboards |

### Dashboard Features

- Live price ticker with sparkline history
- Open positions table with mark-to-market PnL, time held, and variant tags
- Order book imbalance gauge
- Technical indicator panels (RSI, MACD, VWAP, Stochastic)
- Recent trades and signals
- Bankroll, equity, and drawdown display
- Auto-reconnect on disconnect

### Standalone Usage

```bash
# Start mock data generator (no bot needed)
python monitoring/mock_data.py

# In another terminal, view the Rich terminal dashboard
python monitoring/terminal_ui.py

# Or open the web dashboard
open http://localhost:8766
```

## Installation

```bash
# Clone
git clone <repo-url>
cd polymarket-bot

# Virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Dependencies
pip install -r requirements.txt

# Monitoring extras (Rich terminal UI)
pip install rich websockets

# Configure
cp .env.example .env
# Edit .env -- see Configuration Profiles below
```

## Configuration

### Wallet (Required for live trading)

| Variable | Description |
|----------|-------------|
| `PK` | Private key (with 0x prefix) |
| `FUNDER` | Polymarket deposit address (proxy wallets only) |
| `CLOB_API_KEY` | CLOB API key (optional, auto-derived from PK) |
| `CLOB_SECRET` | CLOB API secret |
| `CLOB_PASS_PHRASE` | CLOB API passphrase |

### Configuration Profiles

The `.env.example` file includes three pre-built profiles. Uncomment the one that fits:

#### Conservative Profile
Low risk, small positions, strict protections. Good for starting out.

```env
TRADING_MODE=conservative
BASE_POSITION_SIZE=5
MAX_POSITION_PCT=0.05
MAX_CONCURRENT_POSITIONS=2
DAILY_LOSS_LIMIT=0.10
MAX_DRAWDOWN_PCT=0.10
MAX_CONSECUTIVE_LOSSES=2
COOLOFF_PERIOD_MINUTES=60
PEAK_DECAY_RATE_PER_HOUR=0.03
```

#### Normal Profile (Default)
Balanced risk/reward. Trades all four assets.

```env
TRADING_MODE=normal
BASE_POSITION_SIZE=25
MAX_POSITION_PCT=0.10
MAX_CONCURRENT_POSITIONS=4
DAILY_LOSS_LIMIT=0.25
MAX_DRAWDOWN_PCT=0.15
MAX_CONSECUTIVE_LOSSES=3
COOLOFF_PERIOD_MINUTES=30
PEAK_DECAY_RATE_PER_HOUR=0.05
```

#### Aggressive Profile
Higher risk, larger positions, faster recovery. For experienced users.

```env
TRADING_MODE=aggressive
BASE_POSITION_SIZE=50
MAX_POSITION_PCT=0.15
MAX_CONCURRENT_POSITIONS=6
DAILY_LOSS_LIMIT=0.35
MAX_DRAWDOWN_PCT=0.20
MAX_CONSECUTIVE_LOSSES=5
COOLOFF_PERIOD_MINUTES=15
PEAK_DECAY_RATE_PER_HOUR=0.10
```

### All Configuration Variables

#### Runtime Mode

| Variable | Default | Description |
|----------|---------|-------------|
| `DRY_RUN` | `true` | Simulate without placing orders |
| `PAPER_TRADING_ENABLED` | `false` | Paper trading with simulated fills |
| `PAPER_INITIAL_BALANCE` | `1000` | Starting paper balance (USD) |
| `BOT_STATE_FILE` | `bot_state.json` | Path for state persistence file |
| `LOOP_INTERVAL` | `0.25` | Tick loop interval in seconds |

#### Signal Generation

| Variable | Default | Description |
|----------|---------|-------------|
| `SIMPLE_MODE` | `true` | Distance + trend strategy (recommended) |
| `AGGRESSIVE_MODE` | `false` | Trade every market, always pick a side |
| `TRADING_MODE` | `normal` | Risk profile: `conservative`, `normal`, `aggressive` |
| `CHART_FILTER_ENABLED` | `true` | Use Binance chart analysis to filter signals |

#### Market Selection

| Variable | Default | Description |
|----------|---------|-------------|
| `TRADING_VARIANTS` | `fifteen` | Market durations: `fifteen`, `five`, or `fifteen,five` |
| `ASSET_PRIORITY` | `BTC,ETH,SOL,XRP` | Asset trading priority order |

#### Position Sizing

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_POSITION_SIZE` | `25` | Base position size in USD |
| `MAX_POSITION_PCT` | `0.10` | Max single position as % of bankroll |
| `MAX_CONCURRENT_POSITIONS` | `4` | Max open positions at once |
| `MAX_TOTAL_EXPOSURE_USD` | `200` | Hard cap on total exposure (USD) |
| `MAX_EXPOSURE_PER_ASSET_USD` | `75` | Max exposure per asset (USD) |
| `MAX_TOTAL_EXPOSURE_PCT` | `0.25` | Max total exposure as % of equity |
| `DYNAMIC_SIZING_ENABLED` | `true` | Volatility-adjusted position sizing |
| `TARGET_VOLATILITY` | `0.006` | Target volatility for dynamic sizing |
| `KELLY_CAP` | `0.20` | Maximum Kelly fraction |
| `MIN_TRADE_USD` | `5.0` | Minimum trade size |

#### Edge and Order Types

| Variable | Default | Description |
|----------|---------|-------------|
| `MIN_EDGE` | `0.00` | Minimum edge to take any trade |
| `EDGE_FOR_POST_ONLY` | `0.03` | Min edge for maker orders (earn rebates) |
| `EDGE_FOR_LIMIT` | `0.08` | Min edge for limit orders |
| `EDGE_FOR_MARKET` | `0.15` | Min edge for market orders (pay fees) |

#### Timing

| Variable | Default | Description |
|----------|---------|-------------|
| `MIN_TIME_REMAINING` | `30` | Min seconds before expiry (15-min markets) |
| `MIN_TIME_REMAINING_5M` | `15` | Min seconds before expiry (5-min markets) |
| `MIN_OBSERVATION_TIME` | `30` | Min seconds observing before first trade |
| `MIN_OBSERVATION_TIME_5M` | `10` | Min observation for 5-min markets |
| `TIME_FOR_MARKET` | `60` | Use market orders below this many seconds |
| `TIME_FOR_LIMIT` | `120` | Use limit orders below this many seconds |
| `MAKER_ORDER_TIMEOUT` | `30` | Cancel unfilled maker orders after (seconds) |
| `ORDER_COOLDOWN_SECONDS` | `10` | Cooldown between orders on same asset |
| `WARMUP_PERIOD` | `60` | Data warmup period at startup (seconds) |

#### Loss Protection

| Variable | Default | Description |
|----------|---------|-------------|
| `DAILY_LOSS_LIMIT` | `0.25` | Halt if daily loss exceeds this % |
| `MAX_CONSECUTIVE_LOSSES` | `3` | Enter cooloff after N losses in a row |
| `MAX_DRAWDOWN_PCT` | `0.15` | Enter cooloff if equity drops this % from peak |
| `MIN_WIN_RATE` | `0.40` | Pause if win rate drops below this |
| `MIN_TRADES_FOR_WINRATE` | `5` | Min trades before win rate check activates |
| `COOLOFF_PERIOD_MINUTES` | `30` | Duration of cooloff pause (minutes) |
| `PEAK_DECAY_RATE_PER_HOUR` | `0.05` | Gradual peak decay rate after drawdown cooloff |

#### Early Exit

| Variable | Default | Description |
|----------|---------|-------------|
| `EARLY_EXIT_ENABLED` | `false` | Enable early position exit |
| `TAKE_PROFIT_PCT` | `0.30` | Take profit threshold (+30%) |
| `STOP_LOSS_PCT` | `0.25` | Stop loss threshold (-25%) |
| `EARLY_EXIT_MIN_HOLD` | `60` | Min hold time before exit allowed (seconds) |
| `TIME_EXIT_ENABLED` | `true` | Enable time-based exit |
| `TIME_EXIT_SECONDS` | `90` | Exit positions with less than this remaining |

#### Trend Protection

| Variable | Default | Description |
|----------|---------|-------------|
| `TREND_PROTECTION_ENABLED` | `false` | Block trades against strong trends |
| `MAX_OPPOSITE_TREND_1D` | `0.25` | Block if 1d trend > 25% against signal |
| `MAX_OPPOSITE_TREND_1H` | `0.40` | Block if 1h trend > 40% against signal |
| `VELOCITY_GUARD_ENABLED` | `true` | Block on extreme price velocity |
| `BINANCE_MOMENTUM_ENABLED` | `true` | Use Binance momentum confirmation |

#### Execution

| Variable | Default | Description |
|----------|---------|-------------|
| `SMART_ROUTER_ENABLED` | `true` | Auto-select maker vs taker orders |
| `MAKER_EDGE_THRESHOLD` | `0.02` | Edge threshold for maker routing |
| `TAKER_EDGE_THRESHOLD` | `0.06` | Edge threshold for taker routing |
| `MAX_SPREAD` | `0.05` | Max acceptable bid-ask spread |
| `PARALLEL_EXECUTION` | `true` | Execute orders in parallel |

#### Paper Trading

| Variable | Default | Description |
|----------|---------|-------------|
| `PAPER_MAKER_FEE_BPS` | `1.0` | Simulated maker fee (basis points) |
| `PAPER_TAKER_FEE_BPS` | `2.5` | Simulated taker fee (basis points) |
| `PAPER_SLIPPAGE_BPS` | `2.0` | Simulated slippage (basis points) |

#### WebSocket Resilience

| Variable | Default | Description |
|----------|---------|-------------|
| `WS_MAX_RETRIES` | `10` | Max consecutive reconnection attempts |
| `WS_RECONNECT_DELAY` | `1.0` | Initial reconnect delay (seconds) |
| `WS_MAX_RECONNECT_DELAY` | `60.0` | Max reconnect delay (seconds) |
| `WS_PING_INTERVAL` | `30` | WebSocket ping interval (seconds) |
| `WS_PING_TIMEOUT` | `10` | WebSocket ping timeout (seconds) |

#### Monitoring

| Variable | Default | Description |
|----------|---------|-------------|
| `MONITORING_ENABLED` | `true` | Enable metrics collection |
| `MONITOR_WS_PORT` | `8765` | WebSocket port for dashboard |
| `MONITORING_OUTPUT_PATH` | `data/metrics.jsonl` | Metrics file path |

#### Notifications (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | -- | Telegram bot token for alerts |
| `TELEGRAM_CHAT_ID` | -- | Telegram chat ID |
| `DISCORD_WEBHOOK_URL` | -- | Discord webhook for alerts |

## Project Structure

```
polymarket-bot/
├── src/
│   ├── main.py                  # Entry point
│   ├── bot.py                   # Main orchestrator (~6000 lines)
│   ├── config.py                # All configuration
│   ├── models.py                # MarketState, Position, Signal, OrderBook
│   ├── state.py                 # State persistence (JSON, atomic writes)
│   ├── capital_scaling.py       # Dynamic position sizing
│   ├── kill_switch.py           # Emergency stop
│   │
│   ├── data/
│   │   ├── chainlink.py         # Chainlink RTDS price feed (WebSocket)
│   │   ├── clob.py              # Polymarket CLOB order book (WebSocket)
│   │   ├── gamma.py             # Market discovery via Gamma API
│   │   ├── binance.py           # Binance price feed
│   │   └── binance_chart.py     # Binance candlestick analysis
│   │
│   ├── execution/
│   │   ├── client.py            # Trading client (py-clob-client)
│   │   ├── paper.py             # Paper trading executor
│   │   └── orders.py            # Order management
│   │
│   ├── strategy/
│   │   ├── simple_signals.py    # Distance + trend signal strategy
│   │   ├── signals.py           # Signal generation coordinator
│   │   ├── risk.py              # Risk manager (drawdown, cooloff, decay)
│   │   ├── arbitrage.py         # Arbitrage detection
│   │   ├── ml_predictor.py      # ML prediction engine (83 features)
│   │   ├── ml_ensemble.py       # Gradient boosting ensemble
│   │   └── auto_retrain.py      # Automatic model retraining
│   │
│   ├── ml/
│   │   ├── interface.py         # Clean ML interface (MLInput, MLDecision)
│   │   ├── predictor.py         # Production ML implementation
│   │   ├── feature_builder.py   # LightGBM feature schema
│   │   ├── infer.py             # Model inference engine
│   │   ├── train.py             # Training pipeline
│   │   └── dataset.py           # Dataset builder
│   │
│   ├── monitoring/
│   │   └── dashboard.py         # Internal metrics collection
│   │
│   └── utils/
│       └── logging.py           # Structured logging, mode labels
│
├── monitoring/
│   ├── monitor_server.py        # WebSocket + HTTP server
│   ├── terminal_ui.py           # Rich terminal dashboard
│   ├── web_dashboard.html       # Browser dashboard (single file)
│   ├── mock_data.py             # Fake data generator for testing
│   ├── requirements.txt         # rich, websockets, aiohttp
│   └── README.md                # Monitoring-specific docs
│
├── models/                      # Trained ML models
├── bot_state.json               # Persisted bot state (auto-generated)
├── requirements.txt
├── .env.example                 # Configuration template with profiles
└── config.example.env           # Alternative config template
```

## Supported Assets

| Asset | Markets | Typical Volatility |
|-------|---------|-------------------|
| BTC | 15-min, 5-min | 0.25% - 0.40% |
| ETH | 15-min, 5-min | 0.35% - 0.50% |
| SOL | 15-min, 5-min | 0.50% - 0.80% |
| XRP | 15-min, 5-min | 0.40% - 0.70% |

## Usage

```bash
# Paper trading (default, recommended to start)
PAPER_TRADING_ENABLED=true python -m src.main

# Dry run (no orders at all, just signals)
python -m src.main --dry-run

# Live trading
DRY_RUN=false PAPER_TRADING_ENABLED=false python -m src.main

# With debug logging
python -m src.main --log-level DEBUG

# Trade both 5-min and 15-min markets
TRADING_VARIANTS=fifteen,five python -m src.main
```

## Structured Logging

The bot emits structured log lines for every significant event. Key log tags:

| Log Tag | Description |
|---------|-------------|
| `ORDER_DECISION` | Signal evaluation (SKIP or PLACE_ORDER) with variant |
| `ORDER_SUBMIT` | Order sent to exchange |
| `ORDER_RESULT` | Fill confirmation or failure |
| `SETTLE_APPLY` | Position settlement with PnL trail |
| `POSITION_VALUATION` | Mark-to-market update |
| `PEAK_UPDATED` | New equity high watermark |
| `PEAK_DECAY_START` | Gradual peak decay initiated |
| `PEAK_DECAY_RESOLVED` | Drawdown fell below limit |
| `TRADING_PAUSED` / `TRADING_RESUMED` | Risk pause transitions |
| `DATA_HEALTH` | Feed health gate status |
| `STATE_SAVED` | Periodic state snapshot |
| `SHUTDOWN_PHASE` | Graceful shutdown progress |

## Disclaimer

This software is for educational purposes only. Trading cryptocurrency derivatives involves substantial risk of loss. Only trade with funds you can afford to lose. The authors are not responsible for any financial losses incurred through use of this software.

## License

MIT License
