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

### Probability Model

Uses log-normal price dynamics to calculate true probability:

```python
# Scale volatility to remaining time
scaled_vol = volatility_15min * sqrt(time_remaining / 900)

# Calculate z-score
z_score = log(current_price / target_price) / scaled_vol

# True probability
prob_up = norm.cdf(z_score)
```

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
│  Signal Generator      │  Risk Manager         │            │
│  - Probability calc    │  - Position limits    │            │
│  - Edge detection      │  - Daily loss limit   │            │
│  - Action decision     │  - Size adjustment    │            │
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
| `MIN_EDGE` | 0.25 | Minimum edge to trade (25%) |
| `MIN_TIME_REMAINING` | 45 | Min seconds before expiry |
| `BASE_POSITION_SIZE` | 50 | Position size in USD |
| `MAX_POSITION_PCT` | 0.10 | Max position % of bankroll |
| `MAX_CONCURRENT_POSITIONS` | 5 | Max open positions |
| `DAILY_LOSS_LIMIT` | 0.20 | Daily loss limit (20%) |

### Edge and Time Thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `EDGE_FOR_POST_ONLY` | 0.30 | Min edge for maker orders |
| `EDGE_FOR_LIMIT` | 0.40 | Min edge for limit orders |
| `EDGE_FOR_MARKET` | 0.50 | Min edge for market orders |
| `TIME_FOR_MARKET` | 60 | Seconds remaining for market orders |
| `TIME_FOR_LIMIT` | 120 | Seconds remaining for limit orders |
| `MAKER_ORDER_TIMEOUT` | 30 | Maker order timeout (seconds) |

### WebSocket Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `WS_MAX_RETRIES` | 10 | Max reconnect attempts before circuit breaker |
| `WS_RECONNECT_DELAY` | 1.0 | Initial reconnect delay (seconds) |
| `WS_MAX_RECONNECT_DELAY` | 60.0 | Max reconnect delay (seconds) |
| `WS_PING_INTERVAL` | 30 | Keepalive ping interval (seconds) |
| `WS_PING_TIMEOUT` | 10 | Ping timeout (seconds) |

### Optional Notifications

| Variable | Description |
|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Telegram bot token |
| `TELEGRAM_CHAT_ID` | Telegram chat ID |
| `DISCORD_WEBHOOK_URL` | Discord webhook URL |

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

## Order Execution Strategy

The bot uses different order types based on edge strength and time urgency:

| Edge | Time Remaining | Action | Description |
|------|----------------|--------|-------------|
| ≥50% | <60s | MARKET (FOK) | Guaranteed fill |
| ≥40% | <120s | LIMIT (GTC) | May take liquidity |
| ≥30% | >120s | POST_ONLY | Earn maker rebates |
| <30% | - | SKIP | Edge too low |

### Fee Structure

**Taker Fees (avoid when possible):**
```
fee_rate = 0.25 × (price × (1 - price))²
```
- At 50% odds: ~1.56% fee
- At 30% odds: ~0.56% fee
- At 10% odds: ~0.08% fee

**Maker Rebates:**
- 100% of collected taker fees redistributed to makers
- Daily USDC payments proportional to filled maker volume

## Project Structure

```
polymarket-bot/
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── .env.example           # Environment template
├── .gitignore
├── README.md
└── src/
    ├── __init__.py
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
    │   └── risk.py        # Risk management
    └── utils/
        └── logging.py     # Logging utilities
```

## Risk Management

### Built-in Protections

1. **Position Limits**: Max positions and max size per position
2. **Daily Loss Limit**: Automatically halts trading at 20% daily loss
3. **Minimum Time**: Won't trade with less than 45 seconds remaining
4. **Edge Threshold**: Requires 25%+ edge to enter trades

### Connection Resilience

- **Circuit Breaker**: WebSocket feeds automatically reconnect with exponential backoff
- **Max Retries**: After 10 consecutive failures, circuit breaker opens to prevent resource exhaustion
- **Feed Monitoring**: Trading halts if both Chainlink and CLOB feeds fail
- **Thread Safety**: Market operations use async locks to prevent race conditions
- **Non-blocking Notifications**: Trade alerts sent via background thread pool

### Key Principles

- **Chainlink is truth** - Settlement uses Chainlink, trade based on Chainlink
- **Speed wins** - Sub-second reaction to price movements
- **Maker first** - Earn rebates when possible
- **No prediction** - React to confirmed price data, don't guess direction
- **Mechanical execution** - Same logic every trade
- **Small edges compound** - Many 25% edge trades > few 50% edge trades
- **Survive to trade** - Risk management prevents ruin

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

## Logging

Trades are logged to `trades.jsonl` in JSON Lines format:

```json
{
  "timestamp": "2024-01-15T14:30:00.000Z",
  "signal_id": "sig_BTC_UP_20240115143000",
  "market": {"asset": "BTC", "target_price": 97500},
  "signal": {"side": "UP", "edge": 0.32, "true_prob": 0.72},
  "result": {"success": true, "filled_size": 100}
}
```

## Disclaimer

This software is for educational purposes only. Trading cryptocurrency derivatives involves substantial risk of loss. Only trade with funds you can afford to lose. The authors are not responsible for any financial losses incurred through use of this software.

## License

MIT License
