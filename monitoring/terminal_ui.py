#!/usr/bin/env python3
"""
Rich terminal dashboard — connects to MonitorServer via WebSocket.

Usage:
    python monitoring/terminal_ui.py
    python monitoring/terminal_ui.py --port 8765 --host localhost
"""

import argparse
import asyncio
import json
import sys
import threading
import time
from typing import Optional

from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import websockets

# ── colour palette ──────────────────────────────────────────────
C_CYAN = "bright_cyan"
C_GREEN = "green"
C_RED = "red"
C_MAG = "magenta"
C_YELLOW = "yellow"
C_DIM = "dim"
C_WHITE = "white"
C_BULL = "bold bright_green"
C_BEAR = "bold bright_red"


def _col(val, bullish: Optional[bool] = None) -> str:
    """Colour helper: green if bullish, red if bearish, white otherwise."""
    if bullish is True:
        return f"[{C_BULL}]{val}[/]"
    if bullish is False:
        return f"[{C_BEAR}]{val}[/]"
    return f"[{C_WHITE}]{val}[/]"


def _pct(v: float) -> str:
    return f"{v:+.2f}%"


def _usd(v: float) -> str:
    return f"${v:,.2f}"


def _signal_color(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("bullish", "buy", "long", "up")):
        return C_GREEN
    if any(k in t for k in ("bearish", "sell", "short", "down")):
        return C_RED
    return C_YELLOW


# ── panel builders ──────────────────────────────────────────────

def build_header(d: dict) -> Panel:
    symbol = d.get("symbol", "---")
    tf = d.get("timeframe", "--")
    price = d.get("price", 0)
    change_pct = d.get("change_pct", 0)
    high = d.get("high", 0)
    low = d.get("low", 0)
    status = d.get("bot_status", "RUNNING")
    ts = d.get("timestamp", "")
    mode = d.get("mode", "")
    bankroll = d.get("bankroll", 0)
    equity = d.get("equity", 0)

    bull = change_pct >= 0
    parts = Text()
    parts.append(f"  {symbol} ", style=f"bold {C_MAG}")
    if tf:
        parts.append(f" {tf} ", style=f"bold {C_CYAN} on dark_blue")
    parts.append("  ")
    parts.append(f"${price:,.2f} ", style=C_BULL if bull else C_BEAR)
    if change_pct:
        parts.append(f"({change_pct:+.2f}%) ", style=C_GREEN if bull else C_RED)
    if high:
        parts.append(f"  H {_usd(high)}  L {_usd(low)}", style=C_DIM)
    # Account info
    if bankroll:
        parts.append(f"  💰${bankroll:,.2f}", style=C_WHITE)
        if equity and abs(equity - bankroll) > 0.01:
            eq_col = C_GREEN if equity > bankroll else C_RED
            parts.append(f" (eq ${equity:,.2f})", style=eq_col)
    parts.append("    ")
    status_color = C_GREEN if status == "RUNNING" else C_RED if status == "KILLED" else C_YELLOW
    parts.append(f"[{status}]", style=f"bold {status_color}")
    if mode:
        mode_color = C_YELLOW if "paper" in mode.lower() else C_GREEN
        parts.append(f" {mode.upper()}", style=mode_color)
    parts.append(f"  {ts}", style=C_DIM)

    return Panel(parts, style=C_CYAN, height=3)


def build_orderbook(d: dict) -> Panel:
    obi = d.get("obi", 0)
    obi_signal = d.get("obi_signal", "NEUTRAL")
    depth_01 = d.get("depth_01", 0)
    depth_05 = d.get("depth_05", 0)
    depth_10 = d.get("depth_10", 0)
    buy_walls = d.get("buy_walls", [])
    sell_walls = d.get("sell_walls", [])

    bull = obi > 0
    t = Table.grid(padding=(0, 1))
    t.add_column(style=C_CYAN, width=14)
    t.add_column(width=24)

    sig_col = C_GREEN if "BULL" in obi_signal.upper() else C_RED if "BEAR" in obi_signal.upper() else C_YELLOW
    t.add_row("OBI", f"[bold {sig_col}]{obi:+.1f}%  {obi_signal}[/]")
    t.add_row("Depth 0.1%", f"[{C_WHITE}]{depth_01:,.0f}[/]")
    t.add_row("Depth 0.5%", f"[{C_WHITE}]{depth_05:,.0f}[/]")
    t.add_row("Depth 1.0%", f"[{C_WHITE}]{depth_10:,.0f}[/]")

    if buy_walls:
        walls_str = ", ".join(_usd(w) for w in buy_walls[:3])
        t.add_row(f"[{C_GREEN}]Buy Walls[/]", f"[{C_GREEN}]{walls_str}[/]")
    if sell_walls:
        walls_str = ", ".join(_usd(w) for w in sell_walls[:3])
        t.add_row(f"[{C_RED}]Sell Walls[/]", f"[{C_RED}]{walls_str}[/]")

    return Panel(t, title=f"[{C_MAG}]ORDER BOOK[/]", border_style=C_DIM, padding=(0, 1))


def build_technical(d: dict) -> Panel:
    rsi = d.get("rsi_14", 0)
    macd = d.get("macd", 0)
    macd_hist = d.get("macd_histogram", 0)
    vwap = d.get("vwap", 0)
    price = d.get("price", 0)
    ema5 = d.get("ema_5", 0)
    ema20 = d.get("ema_20", 0)
    ha_trend = d.get("ha_trend", "FLAT")

    t = Table.grid(padding=(0, 1))
    t.add_column(style=C_CYAN, width=14)
    t.add_column(width=24)

    # RSI
    rsi_color = C_RED if rsi > 70 else C_GREEN if rsi < 30 else C_WHITE
    rsi_label = "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else ""
    t.add_row("RSI(14)", f"[{rsi_color}]{rsi:.1f} {rsi_label}[/]")

    # MACD
    macd_color = C_GREEN if macd_hist > 0 else C_RED
    arrow = "▲" if macd_hist > 0 else "▼"
    t.add_row("MACD", f"[{macd_color}]{macd:.2f} {arrow}[/]")
    t.add_row("Histogram", f"[{macd_color}]{macd_hist:.2f}[/]")

    # VWAP
    vwap_pos = "ABOVE" if price > vwap else "BELOW"
    vwap_col = C_GREEN if price > vwap else C_RED
    t.add_row("VWAP", f"[{C_WHITE}]{_usd(vwap)}[/] [{vwap_col}]{vwap_pos}[/]")

    # EMA crossover
    ema_cross = "GOLDEN CROSS" if ema5 > ema20 else "DEATH CROSS"
    ema_col = C_GREEN if ema5 > ema20 else C_RED
    t.add_row("EMA 5/20", f"[{ema_col}]{ema_cross}[/]")
    t.add_row("", f"[{C_DIM}]{_usd(ema5)} / {_usd(ema20)}[/]")

    # Heikin Ashi
    ha_col = C_GREEN if "BULL" in ha_trend.upper() or "UP" in ha_trend.upper() else C_RED if "BEAR" in ha_trend.upper() or "DOWN" in ha_trend.upper() else C_YELLOW
    t.add_row("Heikin Ashi", f"[{ha_col}]{ha_trend}[/]")

    return Panel(t, title=f"[{C_MAG}]TECHNICAL[/]", border_style=C_DIM, padding=(0, 1))


def _volume_bar(pct: float, width: int = 16) -> str:
    """Render a horizontal bar using block characters."""
    filled = int(abs(pct) / 100 * width)
    filled = min(filled, width)
    blocks = "█" * filled + "▓" * min(1, width - filled) + "░" * max(0, width - filled - 1)
    color = C_GREEN if pct >= 0 else C_RED
    return f"[{color}]{blocks[:width]}[/]"


def build_flow(d: dict) -> Panel:
    cvd_1m = d.get("cvd_1m", 0)
    cvd_3m = d.get("cvd_3m", 0)
    cvd_5m = d.get("cvd_5m", 0)
    delta_1m = d.get("delta_1m", 0)
    poc = d.get("poc", 0)
    vol_profile = d.get("volume_profile", [])

    t = Table.grid(padding=(0, 1))
    t.add_column(style=C_CYAN, width=14)
    t.add_column(width=24)

    cvd_col = lambda v: C_GREEN if v > 0 else C_RED
    t.add_row("CVD 1m", f"[{cvd_col(cvd_1m)}]{cvd_1m:+,.0f}[/]")
    t.add_row("CVD 3m", f"[{cvd_col(cvd_3m)}]{cvd_3m:+,.0f}[/]")
    t.add_row("CVD 5m", f"[{cvd_col(cvd_5m)}]{cvd_5m:+,.0f}[/]")
    t.add_row("Delta 1m", f"[{cvd_col(delta_1m)}]{delta_1m:+,.0f}[/]")
    t.add_row("POC", f"[{C_WHITE}]{_usd(poc)}[/]")

    # Volume profile mini bars
    if vol_profile:
        t.add_row("", "")
        t.add_row(f"[{C_DIM}]Vol Profile[/]", "")
        for level in vol_profile[:6]:
            price_str = f"{level.get('price', 0):>10,.0f}"
            pct = level.get("pct", 0)
            t.add_row(f"[{C_DIM}]{price_str}[/]", _volume_bar(pct))

    return Panel(t, title=f"[{C_MAG}]FLOW & VOLUME[/]", border_style=C_DIM, padding=(0, 1))


def build_trades(d: dict) -> Panel:
    total_pnl = d.get("total_pnl", 0)
    unrealized = d.get("unrealized_pnl", 0)
    realized = d.get("realized_pnl", total_pnl - unrealized)
    win_rate = d.get("win_rate", 0)
    total_trades = d.get("total_trades", 0)
    positions = d.get("open_positions", [])
    recent = d.get("recent_trades", [])

    t = Table.grid(padding=(0, 1))
    t.add_column(style=C_CYAN, width=14)
    t.add_column(width=24)

    pnl_col = C_GREEN if total_pnl >= 0 else C_RED
    t.add_row("Total PnL", f"[bold {pnl_col}]{_usd(total_pnl)}[/]")
    t.add_row("Unrealized", _col(_usd(unrealized), unrealized >= 0))
    t.add_row("Realized", _col(_usd(realized), realized >= 0))
    wr_col = C_GREEN if win_rate >= 50 else C_RED
    t.add_row("Win Rate", f"[{wr_col}]{win_rate:.1f}%[/]")
    t.add_row("Trades", f"[{C_WHITE}]{total_trades}[/]")

    # Open positions table
    if positions:
        t.add_row("", "")
        pos_tbl = Table(show_header=True, header_style=C_DIM, box=None, padding=(0, 1))
        pos_tbl.add_column("Sym", style=C_WHITE, width=5)
        pos_tbl.add_column("Side", width=5)
        pos_tbl.add_column("Entry", style=C_DIM, width=6)
        pos_tbl.add_column("Mark", style=C_DIM, width=6)
        pos_tbl.add_column("PnL", width=8)
        pos_tbl.add_column("Time", style=C_DIM, width=5)

        for p in positions[:5]:
            sym = p.get("symbol", "?")
            side = p.get("side", "?")
            entry = p.get("entry_price", 0)
            mark = p.get("mark")
            pnl_val = p.get("pnl", 0)
            pnl_pct = p.get("pnl_pct", 0)
            time_rem = p.get("time_remaining")
            variant = p.get("variant", "")
            side_style = C_GREEN if side in ("LONG", "UP") else C_RED
            pnl_style = C_GREEN if pnl_val >= 0 else C_RED
            mark_str = f"{mark:.2f}" if mark is not None else "—"
            time_str = f"{int(time_rem)}s" if time_rem is not None else "—"
            label = f"{sym}"
            if variant == "five":
                label += "[dim]:5m[/]"
            pos_tbl.add_row(
                label,
                f"[{side_style}]{side}[/]",
                f"{entry:.2f}",
                mark_str,
                f"[{pnl_style}]{_usd(pnl_val)} ({pnl_pct:+.1f}%)[/]",
                time_str,
            )
        t.add_row("Positions", pos_tbl)

    # Recent trades
    if recent:
        t.add_row("", "")
        for tr in recent[:5]:
            side = tr.get("side", "?")
            price_val = tr.get("price", 0)
            pnl_val = tr.get("pnl", 0)
            side_col = C_GREEN if side in ("BUY", "LONG") else C_RED
            pnl_color = C_GREEN if pnl_val >= 0 else C_RED
            t.add_row(
                f"[{side_col}]{side}[/]",
                f"[{C_DIM}]@{_usd(price_val)}[/] [{pnl_color}]{_usd(pnl_val)}[/]",
            )

    return Panel(t, title=f"[{C_MAG}]TRADES & PnL[/]", border_style=C_DIM, padding=(0, 1))


def build_signals(d: dict) -> Panel:
    signals = d.get("signals", [])
    if not signals:
        text = Text("  No active signals", style=C_DIM)
        return Panel(text, title=f"[{C_MAG}]SIGNALS[/]", border_style=C_DIM, height=5, padding=(0, 1))

    parts = Text()
    for sig in signals[:8]:
        color = _signal_color(sig)
        parts.append(f"  {sig}\n", style=color)

    return Panel(parts, title=f"[{C_MAG}]SIGNALS[/]", border_style=C_DIM, padding=(0, 1))


# ── layout assembly ─────────────────────────────────────────────

def build_layout(data: dict, connected: bool) -> Layout:
    layout = Layout()

    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="signals", size=7),
        Layout(name="status", size=1),
    )

    layout["body"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )

    layout["header"].update(build_header(data))

    layout["left"].split_column(
        Layout(build_orderbook(data), name="ob"),
        Layout(build_technical(data), name="tech"),
    )

    layout["right"].split_column(
        Layout(build_flow(data), name="flow"),
        Layout(build_trades(data), name="trades"),
    )

    layout["signals"].update(build_signals(data))

    conn_dot = f"[bold {C_GREEN}]●[/] CONNECTED" if connected else f"[bold {C_RED}]●[/] DISCONNECTED"
    status_text = Text.from_markup(f"  {conn_dot}  [dim]ws://localhost  |  Press Ctrl+C to exit[/]")
    layout["status"].update(status_text)

    return layout


# ── websocket client ────────────────────────────────────────────

class DashboardClient:
    """Async WebSocket client that feeds data into the Rich display."""

    def __init__(self, host: str, port: int):
        self.uri = f"ws://{host}:{port}"
        self.data: dict = {}
        self.connected = False
        self._stop = False

    async def run(self):
        backoff = 1.0
        while not self._stop:
            try:
                async with websockets.connect(self.uri, ping_interval=20, ping_timeout=10) as ws:
                    self.connected = True
                    backoff = 1.0
                    async for msg in ws:
                        try:
                            self.data = json.loads(msg)
                        except json.JSONDecodeError:
                            pass
            except (ConnectionRefusedError, OSError, websockets.ConnectionClosed):
                self.connected = False
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, 15.0)
            except Exception:
                self.connected = False
                await asyncio.sleep(2)

    def stop(self):
        self._stop = True


# ── main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Trading Bot Terminal Dashboard")
    parser.add_argument("--host", default="localhost", help="MonitorServer host")
    parser.add_argument("--port", type=int, default=8765, help="MonitorServer WebSocket port")
    args = parser.parse_args()

    client = DashboardClient(args.host, args.port)
    console = Console()

    # Run WebSocket client in a background thread
    ws_loop = asyncio.new_event_loop()
    ws_thread = threading.Thread(target=lambda: ws_loop.run_until_complete(client.run()), daemon=True)
    ws_thread.start()

    try:
        with Live(console=console, refresh_per_second=2, screen=True) as live:
            while True:
                layout = build_layout(client.data, client.connected)
                live.update(layout)
                time.sleep(0.5)
    except KeyboardInterrupt:
        client.stop()
        console.print(f"\n[{C_DIM}]Dashboard closed.[/]")


if __name__ == "__main__":
    main()
