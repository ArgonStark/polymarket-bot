#!/usr/bin/env python3
"""
Mock data generator — pushes realistic fake trading data to MonitorServer.

Usage:
    python monitoring/mock_data.py
    python monitoring/mock_data.py --port 8765

Open the terminal UI or web dashboard to see live data.
"""

import argparse
import math
import random
import time
from datetime import datetime, timezone

from monitor_server import MonitorServer

# ── price simulation ────────────────────────────────────────────

class PriceSim:
    def __init__(self, start=67200.0, volatility=0.0003):
        self.price = start
        self.vol = volatility
        self.high = start
        self.low = start
        self.open = start
        self.tick = 0

    def step(self) -> float:
        drift = math.sin(self.tick * 0.02) * 0.0001
        shock = random.gauss(0, self.vol)
        self.price *= (1 + drift + shock)
        self.high = max(self.high, self.price)
        self.low = min(self.low, self.price)
        self.tick += 1
        return self.price


class IndicatorSim:
    """Slowly oscillating indicators that look realistic."""

    def __init__(self):
        self.rsi = 50.0
        self.macd = 0.0
        self.macd_hist = 0.0
        self.obi = 0.0
        self.cvd = 0.0
        self.t = 0

    def step(self, price: float, prev_price: float):
        self.t += 1
        dp = (price - prev_price) / prev_price if prev_price else 0

        # RSI mean-reverts around 50 with momentum nudges
        self.rsi += dp * 800 + (50 - self.rsi) * 0.02 + random.gauss(0, 1.5)
        self.rsi = max(10, min(90, self.rsi))

        # MACD follows price momentum
        self.macd += dp * 5000 - self.macd * 0.05 + random.gauss(0, 8)
        self.macd_hist += dp * 3000 - self.macd_hist * 0.08 + random.gauss(0, 5)

        # OBI oscillates
        self.obi += random.gauss(0, 3) - self.obi * 0.03
        self.obi = max(-60, min(60, self.obi))

        # CVD trends
        self.cvd += random.gauss(0, 8000) + dp * 200000 - self.cvd * 0.005
        return self


# ── trade simulation ────────────────────────────────────────────

class TradeSim:
    def __init__(self):
        self.total_pnl = 0.0
        self.unrealized = 0.0
        self.total_trades = 0
        self.wins = 0
        self.positions = []
        self.recent_trades = []

    def maybe_open(self, price: float):
        if len(self.positions) >= 3:
            return
        if random.random() < 0.03:  # ~3% chance per tick
            side = random.choice(["LONG", "SHORT"])
            self.positions.append({
                "symbol": random.choice(["BTC", "ETH", "SOL", "XRP"]),
                "side": side,
                "entry_price": price * (1 + random.uniform(-0.002, 0.002)),
                "size": round(random.uniform(0.01, 0.5), 4),
            })

    def maybe_close(self, price: float):
        if not self.positions:
            return
        if random.random() < 0.05:
            pos = self.positions.pop(0)
            side_mult = 1 if pos["side"] == "LONG" else -1
            pnl = (price - pos["entry_price"]) * pos["size"] * side_mult
            self.total_pnl += pnl
            self.total_trades += 1
            if pnl > 0:
                self.wins += 1
            self.recent_trades.insert(0, {
                "side": "SELL" if pos["side"] == "LONG" else "BUY",
                "price": price,
                "pnl": round(pnl, 2),
                "symbol": pos["symbol"],
            })
            self.recent_trades = self.recent_trades[:10]

    def update_unrealized(self, price: float):
        self.unrealized = 0
        for p in self.positions:
            mult = 1 if p["side"] == "LONG" else -1
            p["pnl_pct"] = round(((price / p["entry_price"]) - 1) * 100 * mult, 2)
            self.unrealized += (price - p["entry_price"]) * p.get("size", 0.1) * mult

    @property
    def win_rate(self):
        return (self.wins / self.total_trades * 100) if self.total_trades else 0


# ── signal generation ───────────────────────────────────────────

def make_signals(ind: IndicatorSim, price: float, vwap: float, ema5: float, ema20: float) -> list[str]:
    sigs = []
    if ind.obi > 15:
        sigs.append(f"OBI -> BULLISH (+{ind.obi:.1f}%)")
    elif ind.obi < -15:
        sigs.append(f"OBI -> BEARISH ({ind.obi:.1f}%)")
    if ind.cvd < -50000:
        sigs.append("CVD 5m -> sell pressure")
    elif ind.cvd > 50000:
        sigs.append("CVD 5m -> buy pressure")
    if ema5 > ema20:
        sigs.append("EMA -> golden cross")
    else:
        sigs.append("EMA -> death cross")
    if ind.rsi > 70:
        sigs.append(f"RSI -> OVERBOUGHT ({ind.rsi:.1f})")
    elif ind.rsi < 30:
        sigs.append(f"RSI -> OVERSOLD ({ind.rsi:.1f})")
    if price > vwap * 1.002:
        sigs.append("Price -> above VWAP (bullish)")
    elif price < vwap * 0.998:
        sigs.append("Price -> below VWAP (bearish)")
    if ind.macd_hist > 20:
        sigs.append("MACD -> bullish momentum")
    elif ind.macd_hist < -20:
        sigs.append("MACD -> bearish momentum")
    return sigs[:8]


def make_volume_profile(price: float) -> list[dict]:
    levels = []
    base = round(price / 100) * 100
    for i in range(-3, 4):
        p = base + i * 50
        pct = random.uniform(20, 100) * (1 if random.random() > 0.4 else -1)
        levels.append({"price": p, "pct": round(pct, 1)})
    return levels


# ── main loop ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Mock data generator for MonitorServer")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    args = parser.parse_args()

    server = MonitorServer(port=args.port)
    server.start()

    print(f"\n  Mock data generator running")
    print(f"  WebSocket:  ws://localhost:{server.ws_port}")
    print(f"  Dashboard:  http://localhost:{server.http_port}")
    print(f"\n  Pushing data every 1s... (Ctrl+C to quit)\n")

    sim = PriceSim(start=67200.0)
    ind = IndicatorSim()
    trades = TradeSim()
    vwap = sim.price
    ema5 = sim.price
    ema20 = sim.price
    poc = sim.price

    try:
        while True:
            prev = sim.price
            price = sim.step()
            ind.step(price, prev)

            # Smooth indicators
            vwap += (price - vwap) * 0.01 + random.gauss(0, 2)
            ema5 += (price - ema5) * 0.15
            ema20 += (price - ema20) * 0.05
            poc += (price - poc) * 0.005 + random.gauss(0, 5)

            trades.maybe_open(price)
            trades.maybe_close(price)
            trades.update_unrealized(price)

            obi_signal = "BULLISH" if ind.obi > 10 else "BEARISH" if ind.obi < -10 else "NEUTRAL"
            ha_trend = "BULLISH" if price > ema5 and ema5 > ema20 else "BEARISH" if price < ema5 and ema5 < ema20 else "FLAT"

            data = {
                "symbol": "BTC",
                "timeframe": "5m",
                "price": round(price, 2),
                "change_pct": round((price / sim.open - 1) * 100, 3),
                "high": round(sim.high, 2),
                "low": round(sim.low, 2),
                "bot_status": "RUNNING",
                "timestamp": datetime.now(timezone.utc).isoformat(),

                # Order Book
                "obi": round(ind.obi, 1),
                "obi_signal": obi_signal,
                "depth_01": int(random.uniform(50000, 200000)),
                "depth_05": int(random.uniform(200000, 800000)),
                "depth_10": int(random.uniform(500000, 2000000)),
                "buy_walls": sorted([round(price - random.uniform(50, 200), 2) for _ in range(2)]),
                "sell_walls": sorted([round(price + random.uniform(50, 200), 2) for _ in range(2)]),

                # Technical
                "rsi_14": round(ind.rsi, 1),
                "macd": round(ind.macd, 2),
                "macd_histogram": round(ind.macd_hist, 2),
                "vwap": round(vwap, 2),
                "ema_5": round(ema5, 2),
                "ema_20": round(ema20, 2),
                "ha_trend": ha_trend,

                # Flow
                "cvd_1m": round(ind.cvd * 0.3, 2),
                "cvd_3m": round(ind.cvd * 0.7, 2),
                "cvd_5m": round(ind.cvd, 2),
                "delta_1m": round(ind.cvd * 0.2 + random.gauss(0, 3000), 2),
                "poc": round(poc, 2),
                "volume_profile": make_volume_profile(price),

                # Signals
                "signals": make_signals(ind, price, vwap, ema5, ema20),

                # Trades
                "total_pnl": round(trades.total_pnl, 2),
                "unrealized_pnl": round(trades.unrealized, 2),
                "realized_pnl": round(trades.total_pnl - trades.unrealized, 2),
                "win_rate": round(trades.win_rate, 1),
                "total_trades": trades.total_trades,
                "open_positions": trades.positions[:5],
                "recent_trades": trades.recent_trades[:5],
            }

            server.push(data)
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping...")
        server.stop()


if __name__ == "__main__":
    main()
