# Polymarket Crypto Arbitrage Bot

An autonomous trading bot for Polymarket's short-duration cryptocurrency prediction markets. Trades **15-minute** and **5-minute** binary outcome markets on BTC, ETH, SOL, and XRP using distance-from-target pricing, multi-timeframe trend analysis, and optional ML filtering — backed by a full risk management stack, real-time monitoring, and crash-safe state persistence.

## Table of Contents

- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Trading Strategy](#trading-strategy)
- [Market Variants](#market-variants)
- [Data Feeds](#data-feeds)
- [Execution Engine](#execution-engine)
- [Risk Management](#risk-management)
- [Machine Learning System](#machine-learning-system)
- [Backtesting](#backtesting)
- [Monitoring & Dashboards](#monitoring--dashboards)
- [State Persistence & Crash Recovery](#state-persistence--crash-recovery)
- [Structured Logging](#structured-logging)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Supported Assets](#supported-assets)
- [Usage](#usage)
- [Tests](#tests)
- [Disclaimer](#disclaimer)
- [License](#license)

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/ArgonStark/polymarket-bot.git
cd polymarket-bot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your wallet credentials and preferred trading profile

# 3. Run in paper trading mode (recommended first)
PAPER_TRADING_ENABLED=true python -m src.main

# 4. Open the monitoring dashboard
# Terminal UI:  python monitoring/terminal_ui.py
# Web UI:       http://localhost:8766
```

## How It Works

Polymarket offers short-duration crypto prediction markets with binary outcomes: "Will BTC be above $97,250 at 3:15 PM?" Markets settle via Chainlink oracle prices.

The bot:
1. **Discovers** active markets via the Gamma API
2. **Streams** real-time prices from Chainlink (settlement oracle), Polymarket CLOB (orderbook), and Binance (leading indicator)
3. **Generates signals** based on price distance from the target, trend direction, and momentum
4. **Validates** signals through risk checks, trade history filters, chart analysis, and optional ML filtering
5. **Executes** orders via maker (POST_ONLY) or taker (LIMIT/MARKET) routing
6. **Settles** positions automatically using API resolution, local Chainlink determination, or on-chain verification
7. **Persists** all state to disk for crash recovery

## Trading Strategy

### Core Concept

The strategy exploits a simple observation: if an asset's price is far from the target with little time remaining, the outcome is highly predictable. This is essentially **selling short-term volatility in a binary option framework**.

Additionally, Binance price updates lead Chainlink by ~50-500ms, providing an informational edge for fast-moving markets.

### Signal Hierarchy (Simple Mode — Default)

The bot uses a hierarchical decision tree with four tiers:

**1. Distance from Target (Primary — >0.3% deviation)**
- Price well above target &rarr; signal **UP** (85%+ confidence)
- Price well below target &rarr; signal **DOWN** (85%+ confidence)
- Rationale: In 15 minutes, price rarely moves 0.3% to cross the target

**2. Reversal Detection (Secondary — catches bounces)**
- Price below target + RSI oversold (<30) &rarr; signal UP (70% confidence)
- Price above target + RSI overbought (>70) &rarr; signal DOWN (70% confidence)
- Multiple reversal signals stack (+5% confidence boost)
- Momentum contradiction applies a -8% penalty

**3. Moderate Distance (0.1%-0.3%)**
- Trust the distance with RSI confirmation
- Base confidence: 62-72%
- RSI alignment adds +3%, warning subtracts -5%

**4. At Target (<0.1%) — Momentum Tiebreaker**
- RSI extremes dominate the decision
- Strong momentum (>35% threshold) breaks ties
- Smallest position sizes — lowest conviction

### Signal Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Simple** (default) | Distance + trend + momentum hierarchy | Recommended for most users |
| **Aggressive** | Always picks a side, trades every market | Higher volume, lower selectivity |
| **Normal** | Complex multi-indicator (RSI, MACD, Bollinger, divergences) | Legacy, more signal sources |

### Binance Confirmation

Binance updates before Chainlink, acting as a leading indicator:

| Confirmation Level | Condition | Action |
|---|---|---|
| **Strong** | Chainlink already crossed target | No boost needed |
| **Medium** | Chainlink near target + Binance crossed | +2% edge boost |
| **Weak** | Only Binance crossed | No trade (too risky) |

### Edge Thresholds & Order Types

| Edge | Order Type | Timing Condition |
|------|-----------|------------------|
| &ge;3% | POST_ONLY | Any time (earns maker rebates) |
| &ge;8% | LIMIT | When <120s remaining |
| &ge;15% | MARKET | When <60s remaining (urgent) |

### Position Sizing

Kelly-inspired sizing with graduated confidence:

```
size = base_size x (1 + edge_factor)
edge_factor = min(edge, 0.5) / 0.5    # scales 0 to 1
```

| Edge | Multiplier | Example ($25 base) |
|------|-----------|-------------------|
| 0% | 1.0x | $25 |
| 5% | 1.25x | $31.25 |
| 25% | 2.0x | $50 |
| 50%+ | 2.0x (capped) | $50 |

Additional adjustments:
- Confidence-based: 80%+ win prob &rarr; 50% of base, <60% &rarr; 20%
- Time decay: final 2 minutes &rarr; 0.7x multiplier
- Negative edge: 0.4x multiplier

Hard caps: 10% of bankroll per trade, $75 per asset, $200 total exposure, max 4 concurrent positions.

### Entry Gate (All Must Pass)

| # | Gate | Requirement |
|---|------|-------------|
| 1 | Time | &ge;30s remaining (15s for 5-min) |
| 2 | Observation | Watched market &ge;30s with &ge;10 Chainlink samples |
| 3 | Spread | Orderbook spread &le;15% or &le;12c |
| 4 | Data Quality | Valid Chainlink price, target price deviation <5% |
| 5 | Edge | Edge &ge; min_edge (0% default) |
| 6 | Position | No existing position for this asset+variant |
| 7 | Risk | Passes all risk manager checks |
| 8 | Win Rate | Asset/side has >35% win rate or <5 trades history |
| 9 | Chart Filter | Trade doesn't contradict strong chart signal (>70%) |
| 10 | ML Filter | Model confidence &ge;52% (if ML enabled) |
| 11 | Balance | Available cash &ge; order size + 10% buffer |
| 12 | Cooldown | No order on this asset in last 10s |
| 13 | Kill Switch | Not active |

### Exit Conditions

| Exit Type | Trigger | Default |
|-----------|---------|---------|
| **Settlement** | Market expires, binary payout ($1 or $0) | Always on |
| **Take-Profit** | P&L &ge; +30% | Off by default |
| **Stop-Loss** | P&L &le; -25% | Off by default |
| **Time Exit** | Edge decayed below 1% with <90s remaining | On |
| **Chart Exit** | Trend reversal detected on 15m chart | On (if chart enabled) |
| **DCA** | Chart signals favorable re-entry at better price | On (max 1 average) |

Early exits require minimum hold time (60s for 15-min, 20s for 5-min) and are controlled by `EARLY_EXIT_ENABLED`.

### Arbitrage Detection

The bot includes multi-pattern arbitrage detection:

| Pattern | Trigger | Action |
|---------|---------|--------|
| **Binary Mispricing** | YES + NO prices < $1.00 | Buy both sides, lock spread |
| **Asymmetric Pricing** | One side significantly underpriced | Buy cheap side |
| **Dump Detection** | 15%+ price drop in 3 seconds | Buy oversold token |
| **Hedge Execution** | After initial leg, when opposite &le;95% | Lock in guaranteed profit |

## Market Variants

| Variant | Duration | Timing Params | Notes |
|---------|----------|---------------|-------|
| **15-minute** | 900s | 30s min remaining, 30s observation | Default, more data per market |
| **5-minute** | 300s | 15s min remaining, 10s observation | Faster turnover, tighter windows |
| **Both** | Mixed | Variant-specific | Trades all available markets |

5-minute markets use **1/3 of the 15-minute timing thresholds** (proportional to duration):

| Parameter | 15-min | 5-min |
|-----------|--------|-------|
| Min time remaining | 30s | 15s |
| Min observation | 30s | 10s |
| Market order urgency | <60s | <20s |
| Limit order window | <120s | <40s |
| Maker order timeout | 30s | 10s |
| Early exit min hold | 60s | 20s |

Configure with `TRADING_VARIANTS=fifteen`, `five`, or `fifteen,five`.

### Settlement Rules

- **UP wins**: End price &ge; Start price (per Chainlink oracle)
- **DOWN wins**: End price < Start price (per Chainlink oracle)

Settlement uses a 3-method resolution pipeline:
1. **API Resolution** &mdash; Query Gamma API for `outcomePrices` (handles auto-resolved markets)
2. **Local Chainlink** &mdash; Use expiry price snapshot captured at market end
3. **On-Chain Override** &mdash; Verify against on-chain data; prefers on-chain if API disagrees

## Data Feeds

The bot streams real-time data from four sources:

| Feed | Protocol | Purpose | Reconnect |
|------|----------|---------|-----------|
| **Chainlink RTDS** | WebSocket | Oracle settlement price (ground truth) | Exponential backoff, circuit breaker |
| **Polymarket CLOB** | WebSocket | Live orderbook, best bid/ask, trades | Auto-reconnect with 5s warmup gate |
| **Binance** | REST/WebSocket | Leading price indicator (~50-500ms ahead of Chainlink) | Polling with staleness detection |
| **Gamma API** | REST | Market discovery, target prices, resolution data | Retry with period boundary detection |

### Data Health Gating

The bot blocks trading when data is stale or unreliable:

- **CLOB orderbook freshness**: Must be updated within 60s
- **Chainlink staleness**: Must be updated within 120s
- **Binance staleness**: Must be updated within 30s
- **CLOB warmup**: 5s mandatory wait after WebSocket reconnect
- **Circuit breakers**: Auto-trip after consecutive feed failures
- **Connection state**: All feeds must report connected

When data health fails, the bot continues running expiry checks and settlements but pauses new trade entry.

### Period Boundary Detection

At the exact moment a new 15/5-minute period starts, the bot captures the current Chainlink price and injects it as the target price for the new period's markets — enabling trading before the Gamma API publishes the new market.

## Execution Engine

### Three Execution Modes

| Mode | Config | Description |
|------|--------|-------------|
| **Dry Run** | `DRY_RUN=true` | Logs signals only, no orders |
| **Paper Trading** | `PAPER_TRADING_ENABLED=true` | Simulated fills with realistic fees/slippage |
| **Live Trading** | Both `false` | Real orders via py-clob-client-v2 |

### Smart Order Routing

When `SMART_ROUTER_ENABLED=true`, the bot auto-selects order type:

| Condition | Route | Rationale |
|-----------|-------|-----------|
| High edge + time pressure | MARKET (FOK) | Guaranteed fill |
| Moderate edge | LIMIT (GTC) | May get better price |
| Low edge + plenty of time | POST_ONLY | Earn maker rebates |

### Paper Trading Simulator

Full order lifecycle simulation with:
- Configurable maker/taker fees (default: 1.0/2.5 bps)
- Configurable slippage (default: 2.0 bps)
- Quote cache with LRU eviction (200 entries)
- Realistic fill simulation (maker vs taker behavior)
- Settlement credit tracking (keeps paper balance in sync)
- Thread-safe order tracking

### Order Lifecycle

```
SIGNAL → VALIDATE → PLACE → PENDING → FILLED → MONITOR → EXIT/SETTLE → CLOSED
```

Features:
- **Order safety guard**: Validates size, price, token before submission
- **Stale order cancellation**: Unfilled maker orders cancelled after timeout
- **Order cooldown**: 10s minimum between orders on same asset
- **Parallel execution**: Multiple orders can execute simultaneously
- **In-flight tracking**: Pending orders tracked for shutdown safety

## Risk Management

### 6+ Protection Layers

| Protection | Default | Behavior |
|------------|---------|----------|
| **Daily Loss Limit** | 25% | Halts trading for the rest of the day |
| **Consecutive Losses** | 3 | Enters 30-minute cooloff |
| **Max Drawdown** | 15% from peak | Enters cooloff, then gradual peak decay |
| **Win Rate Floor** | 40% (after 5 trades) | Pauses until rate recovers |
| **Kill Switch** | Enabled | Hard stop on catastrophic loss |
| **Data Health Gate** | Always on | Blocks trading on stale/missing feeds |
| **CLOB Warmup** | 5 seconds | No trading after WebSocket reconnect |
| **Exposure Caps** | Per-trade, per-asset, total | Hard caps on capital at risk |
| **Trade History Filter** | 35% min win rate | Blocks losing asset/side combos |
| **Chart Filter** | 70% strength | Blocks signals contradicting strong trends |

### Pause System

Consolidated pause management with typed reasons and optional expiry:

- `TRADING_PAUSED` / `TRADING_RESUMED` log lines at every state transition
- Multiple pause reasons can be active simultaneously
- Timed pauses auto-expire (cooloff periods)
- Daily reset clears day-specific pauses at UTC midnight

### Drawdown Recovery (Peak Decay)

When drawdown exceeds the limit:

1. **Cooloff period** (default 30 min) &mdash; no trading, wait for conditions to improve
2. **Peak decay** begins if drawdown still breached after cooloff
   - Peak bankroll exponentially decays toward current bankroll
   - Default rate: 5% of the gap per hour
   - A 20% drawdown resolves below the 15% limit in ~15 hours
   - Decay stops immediately if equity hits a new high
3. **Auto-recovery** if drawdown drops below threshold naturally

### Kill Switch

Multi-trigger emergency stop:

| Trigger | Condition | Recovery |
|---------|-----------|----------|
| Drawdown | Exceeds threshold | Auto-recovers when drawdown falls |
| Daily Loss | Exceeds daily limit | Manual reset or next day |
| Consecutive Losses | Exceeds streak limit | Manual reset |
| File Flag | `KILL_SWITCH` file exists | Remove file |
| Env Var | `KILL_SWITCH=1` | Change env var |

Kill switch state persists across restarts via `bot_state.json`.

### Exposure Management

| Cap | Default | Scope |
|-----|---------|-------|
| Per-trade | 10% of bankroll | Single order |
| Per-asset | $75 USD | All positions in BTC/ETH/SOL/XRP |
| Total (USD) | $200 | All open positions combined |
| Total (%) | 25% of equity | Cash + mark-to-market positions |

Total exposure uses equity (cash + unrealized P&L), not just cash, to prevent over-allocation after wins.

## Machine Learning System

Optional ML layer that filters and enhances the base signal strategy.

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
| Technical indicators | 29 | MACD, Bollinger, Stochastic, RSI divergence, Volume, VWAP |

### Model Architecture

| Model | Type | Features | Role |
|-------|------|----------|------|
| **LightGBM Ensemble** | Gradient Boosting | 83 | Primary predictor, multiple seeds |
| **Random Forest** | Pre-trained | 11 | Baseline from successful trader data |
| **Isotonic Calibration** | Post-hoc | 1 | Calibrates raw probabilities |
| **Hybrid Predictor** | Adaptive blend | All | Weights shift RF&rarr;GBM as samples grow |

### EV-Based ML Engine

The newer ML approach uses expected value calculation:

```
EV = predicted_probability - cost (including fees + slippage)
```

Policy decides based on:
- Minimum EV threshold (1% default)
- Maximum uncertainty (10% default)
- Depth requirements (50+ orders on each side)

Can override the signal generator with its own direction if positive EV exceeds threshold.

### ML Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ML_ENABLED` | `false` | Enable ML confidence filtering |
| `ML_MIN_CONFIDENCE` | `0.52` | Min predicted win probability |
| `ML_MIN_SAMPLES` | `50` | Training samples before ML activates |
| `ML_MIN_EDGE` | `0.02` | Minimum edge override for ML decisions |

### Training Pipeline

```bash
# Train from collected trade data
python scripts/train_from_trader.py

# Check model status
python scripts/ml_status.py
```

Models auto-migrate when feature count changes. Training uses stratified time-based splits to prevent lookahead bias.

## Backtesting

Built-in backtesting engine for strategy validation:

```bash
python scripts/run_backtest.py
```

Components:
- **Engine** (`src/backtest/engine.py`) &mdash; Main backtest loop with realistic simulation
- **Execution** (`src/backtest/execution.py`) &mdash; Simulated order fills with slippage
- **Metrics** (`src/backtest/metrics.py`) &mdash; P&L, Sharpe, drawdown, win rate calculations
- **Data Loader** (`src/backtest/data_loader.py`) &mdash; Historical market data ingestion

## Monitoring & Dashboards

The bot includes a real-time monitoring system that starts automatically.

### Components

| Component | Access | Description |
|-----------|--------|-------------|
| **WebSocket Server** | `ws://localhost:8765` | Broadcasts JSON snapshots to all clients |
| **Web Dashboard** | `http://localhost:8766` | Single-file browser dashboard (dark terminal theme) |
| **Terminal UI** | `python monitoring/terminal_ui.py` | Rich-based terminal dashboard |
| **Metrics JSON** | `src/metrics.json` | Machine-readable metrics snapshot (updated every 30s) |
| **Mock Data** | `python monitoring/mock_data.py` | Generates fake data for testing dashboards |

### Web Dashboard Features

- Real-time price chart (canvas-based, last 600 data points)
- 4 stat cards (Balance, P&L, ROI, Win Rate)
- P&L distribution bar (win/loss ratio visualization)
- Open positions with mark-to-market P&L, variant tags, and countdown timers
- Trade execution log (last 10 trades with WIN/LOSS indicators)
- Live metrics panel (exposure, peak, drawdown, trade stats)
- Animated value changes (green/red flash on updates)
- Auto-reconnect WebSocket with exponential backoff (1s &rarr; 15s cap)
- Terminal/hacker aesthetic (green-on-black, monospace)
- Responsive layout (collapses to single column on mobile)

### Metrics Tracked

| Metric | Description |
|--------|-------------|
| Bankroll / Peak / Drawdown | Current capital, high watermark, distance from peak |
| Realized / Unrealized P&L | Closed trade profits + open position mark-to-market |
| Rolling Win Rate | Last 50 trades sliding window |
| Session Win Rate | Current session statistics |
| Average Edge | Rolling edge of executed trades |
| Edge Decay | Predicted edge vs actual mark (open and closed) |
| Veto Tracking | Blocked trades by reason (5-minute rolling window) |
| On-Chain Stats | Wins/losses/payouts from The Graph |

### Standalone Usage

```bash
# Start mock data generator (no bot needed)
python monitoring/mock_data.py

# In another terminal, view the Rich terminal dashboard
python monitoring/terminal_ui.py

# Or open the web dashboard
open http://localhost:8766
```

## State Persistence & Crash Recovery

Bot state is saved to `bot_state.json` every 60 seconds and on graceful shutdown.

### What's Persisted

| Field | Purpose |
|-------|---------|
| `current_bankroll` | Available cash |
| `peak_bankroll` | High watermark for drawdown calculation |
| `consecutive_losses` | Current losing streak count |
| `trade_history` | Recent trade results (sliding window) |
| `daily_stats` | Today's P&L, trades, wins, losses, fees |
| `open_positions` | All active positions with token IDs |
| `kill_switch` | Active/inactive state, reason, timestamp |
| `settled_markets` | Recently settled market IDs (prevents re-settlement) |
| `execution_mode` | PAPER/DRY/LIVE (prevents mode-switch false drawdown) |

### Safety Features

- **Atomic writes**: tmp file &rarr; `fsync` &rarr; `os.replace` prevents corruption on crash
- **Mode-switch detection**: If saved mode (e.g., PAPER) differs from current mode (LIVE), skip bankroll/peak restoration to prevent false drawdown triggers
- **Orphaned position cleanup**: Positions in markets that expired while bot was offline are force-closed as losses on startup
- **State invariant checks**: Every save verifies paper_executor.balance == risk_manager.current_bankroll
- **Settlement idempotency**: `settled_markets` set prevents double-processing the same market

### Graceful Shutdown (5 Phases)

| Phase | Action | Log Tag |
|-------|--------|---------|
| 1 | Stop tick loops | `SHUTDOWN_PHASE phase=1` |
| 2 | Cancel all pending orders | `SHUTDOWN_PHASE phase=2` |
| 3 | Drain in-flight operations | `SHUTDOWN_PHASE phase=3` |
| 4 | Save state + verify invariants | `SHUTDOWN_PHASE phase=4` |
| 5 | Disconnect feeds and monitoring | `SHUTDOWN_PHASE phase=5` |

Each phase logs elapsed time. The `_running` flag prevents new order submissions during shutdown.

## Structured Logging

The bot emits structured log lines for every significant event:

| Log Tag | Description |
|---------|-------------|
| `ORDER_DECISION` | Signal evaluation (SKIP or PLACE_ORDER) with full context |
| `ORDER_SUBMIT` | Order sent to executor with type, price, size |
| `ORDER_RESULT` | Fill confirmation or failure with execution details |
| `ORDER_BLOCK` | Signal blocked by a gate (with reason) |
| `SETTLE_APPLY` | Position settlement with full PnL audit trail |
| `SETTLE_OVERRIDE` | On-chain outcome overrides API resolution |
| `POSITION_VALUATION` | Mark-to-market update with token_id and price source |
| `PEAK_UPDATED` | New equity high watermark |
| `PEAK_DECAY_START` / `PEAK_DECAY_RESOLVED` | Drawdown recovery lifecycle |
| `TRADING_PAUSED` / `TRADING_RESUMED` | Risk pause state transitions |
| `DATA_HEALTH` | Feed health gate status (PASS/FAIL with reason) |
| `EXPIRY_SNAPSHOT` | Chainlink price captured at market expiry |
| `PERIOD_BOUNDARY` | New market period detected |
| `STATE_SAVED` | Periodic state persistence with invariant check |
| `STATE_DIFF` | State inconsistency detected (logged as ERROR, never crashes) |
| `SHUTDOWN_PHASE` | Graceful shutdown progress with elapsed time |
| `DAILY_RESET` | UTC midnight stats reset |
| `KILL_SWITCH_RESTORED` | Kill switch state loaded from disk |
| `LOCAL_RESOLVE` | Settlement determined via Chainlink snapshot |
| `METRICS_SNAPSHOT` | Periodic metrics dashboard update |

Log throttling prevents spam: `PRICE_LOG` (5s), `SIGNAL_LOG` (30s), `ML_DECISION_LOG` (1s), `ORDER_BLOCK` (per asset:reason, configurable interval).

## Architecture

```
                          DATA LAYER
 +-----------------+-----------------+-----------------+-----------------+
 | Chainlink RTDS  | Polymarket CLOB | Binance REST/WS |   Gamma API     |
 | (Oracle Price)  | (Order Book)    | (Leading Price) | (Market Disc.)  |
 +-------+---------+-------+---------+-------+---------+-------+---------+
         |                 |                 |                 |
         v                 v                 v                 v
 +-----------------------------------------------------------------------+
 |                    TRADING BOT (Mixin Architecture)                    |
 |                                                                       |
 |  TradingMixin      SettlementMixin     EarlyExitMixin                 |
 |  PositionMixin     MonitoringMixin                                    |
 +-----------------------------------------------------------------------+
         |                 |                  |                |
         v                 v                  v                v
 +------------------+  +------------------+  +----------------------------+
 | Signal Generator |  | Risk Manager     |  | ML Engine (83 features)    |
 | Distance + Trend |  | Drawdown/Pause   |  | LightGBM + RF Ensemble    |
 | Reversal Detect  |  | Kill Switch      |  | EV-Based Policy            |
 | Arbitrage Detect |  | Exposure Caps    |  | Isotonic Calibration       |
 +--------+---------+  +--------+---------+  +-------------+--------------+
          |                      |                          |
          v                      v                          v
 +-----------------------------------------------------------------------+
 |                       EXECUTION LAYER                                 |
 |  Paper Executor (simulated)  |  Live Executor (py-clob-client-v2)     |
 |  Smart Router (POST_ONLY / LIMIT / MARKET)                           |
 |  Order Safety Guard  |  Stale Order Cancellation                     |
 +-----------------------------------------------------------------------+
          |                                          |
          v                                          v
 +-----------------------------------------------------------------------+
 |                       MONITORING & STATE                              |
 |  WebSocket Server  |  Web Dashboard  |  Terminal UI (Rich)           |
 |  State Persistence |  Metrics JSON   |  The Graph Integration       |
 +-----------------------------------------------------------------------+
```

### Design Patterns

| Pattern | Implementation |
|---------|---------------|
| **Mixin Architecture** | TradingBot composes 5 mixins (trading, settlement, exits, positions, monitoring) |
| **UP-token canonical** | MarketState stores UP token prices; DOWN derived as complement (1 - price) |
| **Token-ID tracking** | Positions store immutable `yes_token_id`, `no_token_id`, `held_token_id` at open |
| **Atomic state writes** | tmp file + fsync + os.replace for crash-safe persistence |
| **Two-loop architecture** | Trading loop (0.25s tick) + Settlement loop (5s tick) run independently |
| **Three-queue lifecycle** | `markets` (active) &rarr; `expiring_markets` (pending settlement) &rarr; `settled_markets` (archive) |
| **Quote cache isolation** | Keyed by `token_id` (not asset name) to prevent cross-market contamination |

## Installation

```bash
# Clone
git clone https://github.com/ArgonStark/polymarket-bot.git
cd polymarket-bot

# Virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Core dependencies
pip install -r requirements.txt

# Monitoring extras (Rich terminal UI)
pip install rich websockets

# Configure
cp .env.example .env
# Edit .env -- see Configuration Profiles below
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `py-clob-client-v2` | Polymarket CLOB V2 API client |
| `websocket-client` | WebSocket connections (Chainlink, CLOB) |
| `numpy`, `scipy` | Numerical computation |
| `scikit-learn` | ML models (Random Forest, calibration) |
| `lightgbm` | Gradient boosting ensemble |
| `requests`, `aiohttp` | HTTP/async HTTP clients |
| `python-dotenv` | Environment variable configuration |
| `structlog` | Structured logging |
| `rich` (optional) | Terminal UI dashboard |

## Configuration

### Wallet (Required for Live Trading)

| Variable | Description |
|----------|-------------|
| `PK` | Private key (with 0x prefix) |
| `FUNDER` | Polymarket deposit address (proxy wallets only) |
| `CLOB_API_KEY` | CLOB API key (optional, auto-derived from PK) |
| `CLOB_SECRET` | CLOB API secret |
| `CLOB_PASS_PHRASE` | CLOB API passphrase |

### Configuration Profiles

The `.env.example` file includes three pre-built profiles:

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
Higher risk, larger positions, faster recovery.

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

<details>
<summary>Click to expand full configuration reference (60+ variables)</summary>

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

</details>

## Project Structure

```
polymarket-bot/
├── src/
│   ├── main.py                      # Entry point
│   ├── config.py                    # All configuration (60+ variables)
│   ├── models.py                    # MarketState, Position, Signal, OrderBook
│   ├── state.py                     # State persistence (atomic JSON writes)
│   ├── capital_scaling.py           # Dynamic position sizing
│   ├── kill_switch.py               # Emergency stop (multi-trigger)
│   │
│   ├── bot/                         # Main orchestrator (mixin architecture)
│   │   ├── core.py                  # TradingBot class, tick loop, initialize/shutdown
│   │   ├── trading.py               # Signal processing + order execution
│   │   ├── exits.py                 # Early exit logic (TP/SL/time/chart)
│   │   ├── settlement.py            # Settlement processing + on-chain verification
│   │   ├── positions.py             # Position tracking + mark-to-market
│   │   └── monitoring.py            # Snapshot building + status logging
│   │
│   ├── data/                        # Real-time market data feeds
│   │   ├── chainlink.py             # Chainlink RTDS price feed (WebSocket)
│   │   ├── clob.py                  # Polymarket CLOB order book (WebSocket)
│   │   ├── gamma.py                 # Market discovery via Gamma API
│   │   ├── binance.py               # Binance price feed (REST)
│   │   ├── binance_chart.py         # Binance candlestick / technical analysis
│   │   ├── thegraph.py              # The Graph on-chain data
│   │   ├── settlement_verifier.py   # On-chain settlement verification
│   │   └── historical.py            # Historical data loading
│   │
│   ├── execution/                   # Order execution
│   │   ├── client.py                # Live trading client (py-clob-client-v2)
│   │   ├── paper.py                 # Paper trading executor (simulated fills)
│   │   ├── orders.py                # Order management + safety guard
│   │   └── low_latency/
│   │       └── router.py            # Smart order routing
│   │
│   ├── strategy/                    # Signal generation & risk
│   │   ├── simple_signals.py        # Distance + trend signal strategy (default)
│   │   ├── signals.py               # Signal generation coordinator
│   │   ├── aggressive_signals.py    # Always-trade aggressive mode
│   │   ├── unified_signals.py       # Unified signal interface
│   │   ├── risk.py                  # Risk manager (drawdown, cooloff, peak decay)
│   │   ├── risk_sizing.py           # Position size calculation
│   │   ├── kelly.py                 # Kelly criterion sizing
│   │   ├── arbitrage.py             # Arbitrage detection (4 patterns)
│   │   ├── reversal_detector.py     # RSI/momentum reversal detection
│   │   ├── position_monitor.py      # Open position analysis
│   │   ├── averaging.py             # DCA / averaging down logic
│   │   ├── trend_protection.py      # Trend-based trade blocking
│   │   ├── technical_analysis.py    # RSI, MACD, Bollinger, Stochastic, VWAP
│   │   ├── trade_history.py         # Trade tracking and win rate
│   │   ├── ml_ensemble.py           # Gradient boosting ensemble
│   │   ├── meta.py                  # Meta-strategy coordination
│   │   └── backtesting.py           # Strategy backtesting utilities
│   │
│   ├── ml/                          # Machine learning (83 features)
│   │   ├── interface.py             # Clean ML interface (MLInput, MLDecision)
│   │   ├── predictor.py             # Production ML implementation
│   │   ├── feature_builder.py       # LightGBM feature schema
│   │   ├── infer.py                 # Model inference engine
│   │   ├── train.py                 # Training pipeline
│   │   ├── dataset.py               # Dataset builder
│   │   └── policy.py                # EV-based trading policy
│   │
│   ├── monitoring/                  # Metrics collection
│   │   ├── dashboard.py             # Metrics dashboard (JSON snapshots)
│   │   └── metrics.py               # JSONL metrics collector
│   │
│   ├── backtest/                    # Backtesting engine
│   │   ├── engine.py                # Main backtest loop
│   │   ├── execution.py             # Simulated order execution
│   │   ├── metrics.py               # P&L and performance metrics
│   │   ├── models.py                # Backtest data models
│   │   └── data_loader.py           # Historical data ingestion
│   │
│   └── utils/
│       ├── logging.py               # Structured logging, mode labels
│       ├── trading_logger.py        # Trade-specific logging
│       └── console.py               # Console output utilities
│
├── monitoring/                      # External monitoring tools
│   ├── monitor_server.py            # WebSocket + HTTP server (dual-port)
│   ├── terminal_ui.py               # Rich terminal dashboard
│   ├── web_dashboard.html           # Browser dashboard (single file, dark theme)
│   ├── mock_data.py                 # Fake data generator for testing
│   └── requirements.txt             # rich, websockets, aiohttp
│
├── scripts/                         # Utility scripts
│   ├── performance_dashboard.py     # Live P&L tracking
│   ├── ml_status.py                 # ML model status checker
│   ├── train_from_trader.py         # Train ML from trade data
│   ├── run_backtest.py              # Run backtests
│   └── fix_pnl.py                   # P&L correction utility
│
├── tests/                           # 17 test files (pytest)
├── models/                          # Trained ML models
├── requirements.txt                 # Core dependencies
├── .env.example                     # Configuration template (3 profiles)
└── bot_state.json                   # Persisted bot state (auto-generated)
```

## Supported Assets

| Asset | Markets | Typical 15-min Volatility |
|-------|---------|--------------------------|
| BTC | 15-min, 5-min | 0.25% - 0.40% |
| ETH | 15-min, 5-min | 0.35% - 0.50% |
| SOL | 15-min, 5-min | 0.50% - 0.80% |
| XRP | 15-min, 5-min | 0.40% - 0.70% |

Asset priority is configurable: `ASSET_PRIORITY=BTC,ETH,SOL,XRP`

## Usage

```bash
# Paper trading (recommended to start)
PAPER_TRADING_ENABLED=true python -m src.main

# Dry run (signals only, no orders)
python -m src.main --dry-run

# Live trading
DRY_RUN=false PAPER_TRADING_ENABLED=false python -m src.main

# Trade both 5-min and 15-min markets
TRADING_VARIANTS=fifteen,five python -m src.main

# With debug logging
python -m src.main --log-level DEBUG

# Performance dashboard
python scripts/performance_dashboard.py

# Train ML model
python scripts/train_from_trader.py

# Run backtest
python scripts/run_backtest.py
```

## Tests

17 test files covering critical paths:

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_settlement.py -v
```

| Test | Coverage |
|------|----------|
| `test_settlement.py` | Settlement logic, outcome determination |
| `test_settlement_accounting.py` | P&L accounting, paper executor credit |
| `test_state_persistence.py` | JSON save/restore, atomic writes, mode switch |
| `test_paper_executor.py` | Paper trading fills, slippage, fees |
| `test_risk_pause.py` | Pause states, cooloff periods |
| `test_kill_switch.py` | Emergency stop triggers, auto-recovery |
| `test_five_minute_markets.py` | 5-min market variant support |
| `test_clob_warmup.py` | CLOB reconnect warmup gate |
| `test_data_health.py` | Feed health gating |
| `test_exposure_cap.py` | Exposure limits (per-trade, per-asset, total) |
| `test_order_lifecycle.py` | Order state machine |
| `test_capital_scaling.py` | Dynamic position sizing |
| `test_price_cache_collision.py` | Quote cache isolation by token_id |
| `test_stale_order_cancellation.py` | Unfilled order cleanup |
| `test_ml_pipeline.py` | ML training/inference pipeline |
| `test_dashboard.py` | Monitoring dashboard |
| `test_attribution.py` | Trade attribution tracking |

## Disclaimer

This software is for educational purposes only. Trading cryptocurrency derivatives involves substantial risk of loss. Only trade with funds you can afford to lose. The authors are not responsible for any financial losses incurred through use of this software.

## License

MIT License
