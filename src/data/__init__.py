"""Data layer for Chainlink, CLOB, Binance, and market discovery."""

from .chainlink import ChainlinkFeed
from .clob import CLOBFeed
from .gamma import GammaAPI
from .binance import BinanceFeed, BinancePrice
from .polymarket_data import PolymarketDataAPI, get_data_api
from .historical import (
    fetch_historical_prices,
    fetch_all_historical_prices,
    prepopulate_price_histories,
)

__all__ = [
    "ChainlinkFeed",
    "CLOBFeed",
    "GammaAPI",
    "BinanceFeed",
    "BinancePrice",
    "PolymarketDataAPI",
    "get_data_api",
    "fetch_historical_prices",
    "fetch_all_historical_prices",
    "prepopulate_price_histories",
]
