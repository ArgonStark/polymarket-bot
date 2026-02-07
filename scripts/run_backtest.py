#!/usr/bin/env python3
"""Run a backtest over historical orderbook and trade data."""

from __future__ import annotations

import argparse

from src.backtest import BacktestEngine, BacktestConfig, ExecutionConfig, load_orderbooks, load_trades
from src.strategy.risk_sizing import RiskSizingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orderbooks", required=True)
    parser.add_argument("--trades", required=True)
    parser.add_argument("--initial", type=float, default=1000)
    parser.add_argument("--min-edge", type=float, default=0.01)
    parser.add_argument("--exit-edge", type=float, default=0.005)
    parser.add_argument("--exit-seconds", type=int, default=120)
    parser.add_argument("--duration", type=int, default=900)
    return parser.parse_args()


def main():
    args = parse_args()

    orderbooks = load_orderbooks(args.orderbooks)
    trades = load_trades(args.trades)

    engine = BacktestEngine(
        config=BacktestConfig(
            initial_balance=args.initial,
            min_edge=args.min_edge,
            exit_edge=args.exit_edge,
            exit_seconds_before_end=args.exit_seconds,
            market_duration_seconds=args.duration,
        ),
        exec_config=ExecutionConfig(
            maker_fee_bps=1.0,
            taker_fee_bps=2.5,
            latency_ms=50,
            slippage_bps=2.0,
        ),
        risk_config=RiskSizingConfig(
            enabled=True,
            target_volatility=0.006,
            kelly_cap=0.2,
            max_exposure_pct=0.1,
            min_trade_usd=5.0,
        ),
    )

    metrics = engine.run(orderbooks, trades)
    print(metrics)


if __name__ == "__main__":
    main()
