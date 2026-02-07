"""Backtest data loader."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .models import OrderBookSnapshot, TradeTick


def _parse_ts(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def load_orderbooks(path: str) -> list[OrderBookSnapshot]:
    snapshots: list[OrderBookSnapshot] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        ts = _parse_ts(data["ts"])
        bids = [(float(p), float(s)) for p, s in data.get("bids", [])]
        asks = [(float(p), float(s)) for p, s in data.get("asks", [])]
        snapshots.append(
            OrderBookSnapshot(
                ts=ts,
                bids=bids,
                asks=asks,
                reference_price=float(data["reference_price"]) if data.get("reference_price") else None,
            )
        )
    return snapshots


def load_trades(path: str) -> list[TradeTick]:
    trades: list[TradeTick] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        trades.append(
            TradeTick(
                ts=_parse_ts(data["ts"]),
                price=float(data["price"]),
                size=float(data["size"]),
                side=str(data.get("side", "BUY")).upper(),
            )
        )
    return trades


def merge_events(
    orderbooks: Iterable[OrderBookSnapshot],
    trades: Iterable[TradeTick],
) -> list[tuple[datetime, str, object]]:
    events = []
    for ob in orderbooks:
        events.append((ob.ts, "orderbook", ob))
    for tr in trades:
        events.append((tr.ts, "trade", tr))
    events.sort(key=lambda x: x[0])
    return events
