# Monitoring System

Real-time monitoring dashboards for the Polymarket trading bot.

## Architecture

```
Bot (push) ──→ MonitorServer (ws://port:8765)
                  ├──→ Terminal UI (Rich)
                  └──→ Web Dashboard (browser)
```

- **MonitorServer** — WebSocket data hub. Bot pushes JSON snapshots; server broadcasts to all connected clients.
- **Terminal UI** — Rich-based terminal dashboard. Connects as a WebSocket client.
- **Web Dashboard** — Single-file HTML/JS dashboard served by MonitorServer on HTTP port (8766).
- **Mock Data** — Standalone data generator for testing dashboards without the live bot.

## Quick Start

```bash
# Install dependencies
pip install -r monitoring/requirements.txt

# Run with mock data (demo mode)
python monitoring/mock_data.py

# In another terminal — Rich dashboard
python monitoring/terminal_ui.py

# Or open browser dashboard
open http://localhost:8766
```

## Usage with Live Bot

```python
from monitoring.monitor_server import MonitorServer

monitor = MonitorServer(port=8765)
monitor.start()

# In your tick loop:
monitor.push({
    "symbol": "BTC",
    "price": 67145.49,
    "bot_status": "RUNNING",
    # ... full snapshot dict
})

# On shutdown:
monitor.stop()
```

## Components

### `monitor_server.py`

WebSocket server + HTTP server running in a daemon thread.

| Method | Description |
|--------|-------------|
| `start()` | Launch background servers |
| `stop()` | Graceful shutdown |
| `push(data)` | Thread-safe broadcast to all clients |

- WebSocket: `ws://0.0.0.0:8765`
- HTTP (dashboard): `http://0.0.0.0:8766`
- Health check: `http://0.0.0.0:8766/health`

### `terminal_ui.py`

```bash
python monitoring/terminal_ui.py [--host HOST] [--port PORT]
```

Panels: Header, Order Book, Technical, Flow & Volume, Trades & PnL, Signals.

### `web_dashboard.html`

Served automatically by MonitorServer. Features:
- Dark terminal aesthetic with scanline overlay
- Canvas sparkline for price history
- Auto-reconnect with connection indicator
- Flash animations on value changes

### `mock_data.py`

```bash
python monitoring/mock_data.py [--port PORT]
```

Simulates BTC price movement, oscillating indicators, and random trades. Pushes data every 1 second.

## Data Schema

The `push()` dict should contain:

| Field | Type | Description |
|-------|------|-------------|
| `symbol` | str | Asset symbol (e.g. "BTC") |
| `price` | float | Current price |
| `change_pct` | float | Session change % |
| `high` / `low` | float | Session high/low |
| `bot_status` | str | "RUNNING", "PAUSED", etc. |
| `obi` | float | Order book imbalance % |
| `rsi_14` | float | RSI(14) |
| `macd` / `macd_histogram` | float | MACD values |
| `vwap` / `ema_5` / `ema_20` | float | Moving averages |
| `cvd_1m` / `cvd_5m` | float | Cumulative volume delta |
| `total_pnl` | float | Realized PnL |
| `unrealized_pnl` | float | Open position PnL |
| `win_rate` | float | Win rate % |
| `open_positions` | list | Active position dicts |
| `recent_trades` | list | Last N closed trades |
| `signals` | list[str] | Active signal strings |

## Ports

| Service | Default Port | Config |
|---------|-------------|--------|
| WebSocket | 8765 | `--port` flag |
| HTTP Dashboard | 8766 | auto: ws_port + 1 |
