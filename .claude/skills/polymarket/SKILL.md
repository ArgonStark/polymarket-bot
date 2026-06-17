---
name: polymarket
description: Polymarket prediction market platform - CLOB V2 API, py-clob-client-v2 Python SDK, trading bots, order management, market data, websockets, Gamma API. Use when working with Polymarket trading, order placement, market fetching, authentication, conditional tokens, pUSD/USDC, Polygon blockchain integration.
---

# Polymarket

Polymarket is a decentralized prediction market on Polygon where users buy/sell outcome shares (YES/NO) priced 0.00-1.00 (pUSD). Winning shares pay $1.00 on resolution. It uses a Central Limit Order Book (CLOB) for trading via REST API and WebSocket.

## ⚠️ CLOB V2 (live April 28, 2026)

CLOB V2 replaced V1 on production. **Legacy V1 SDKs (`py-clob-client`, `@polymarket/clob-client`) no longer work** — V1-signed orders are rejected and the old GitHub repo was archived May 25, 2026. Use the V2 SDK: `py-clob-client-v2` (PyPI) / `@polymarket/clob-client-v2` (npm).

What changed at cutover:
- **pUSD replaces USDC.e** as the collateral token (standard ERC-20 on Polygon)
- New Exchange contracts (CTF Exchange V2 + Neg Risk CTF Exchange V2); EIP-712 domain version "1" → "2"
- Order struct: **removed** `nonce`, `feeRateBps`, `taker`; **added** `timestamp` (ms), `metadata`, `builder` — fees are now set at match time, not per order
- Native builder attribution via `builder_code`
- All open orders were wiped at cutover; `get_pre_migration_orders()` retrieves old ones

### V1 → V2 Python method renames
| V1 (`py_clob_client`) | V2 (`py_clob_client_v2`) |
|---|---|
| `create_or_derive_api_creds()` | `create_or_derive_api_key()` |
| `get_orders(OpenOrderParams())` | `get_open_orders(params=None)` |
| `cancel(order_id=...)` | `cancel_order(OrderPayload(orderID=...))` |
| `cancel_orders([ids])` | `cancel_orders([order_hashes])` |
| `post_order(o, orderType=...)` | `post_order(o, order_type=..., post_only=False, defer_exec=False)` |
| — | `create_and_post_order(...)`, `create_and_post_market_order(...)` (one-shot helpers) |

`OrderArgs`/`MarketOrderArgs` top-level exports are the V2 structs (no `nonce`/`fee_rate_bps`/`taker`; new optional `builder_code`, `metadata`, `expiration`). `client.mode` (0/1/2), `get_order(order_id)`, `get_trades()`, `cancel_all()`, `get_balance_allowance(BalanceAllowanceParams)`, `BUY`/`SELL` constants, and `constants.POLYGON` are unchanged.

## Quick Start

### Installation

```bash
pip install py-clob-client-v2  # v1.0.1+ (CLOB V2), Python 3.9+
```

### Read-Only Client (no auth)

```python
from py_clob_client_v2.client import ClobClient

client = ClobClient("https://clob.polymarket.com", chain_id=137)  # Level 0
markets = client.get_simplified_markets()
mid = client.get_midpoint("<token-id>")
price = client.get_price("<token-id>", side="BUY")
book = client.get_order_book("<token-id>")
```

### Authenticated Client (trading)

```python
from py_clob_client_v2.client import ClobClient

client = ClobClient(
    "https://clob.polymarket.com",
    key="<private-key>",
    chain_id=137,              # Polygon mainnet (80002 = Amoy testnet)
    signature_type=1,          # 0=EOA, 1=POLY_PROXY (Magic/email), 2=GNOSIS_SAFE, 3=POLY_1271 (deposit wallet)
    funder="<funder-address>"  # Address holding funds (pUSD)
)
client.set_api_creds(client.create_or_derive_api_key())
```

New-style accounts (pUSD deposit wallets) use `SignatureTypeV2.POLY_1271` (3) with the deposit wallet address as `funder`; older Magic/email proxies remain type 1, Gnosis Safe type 2.

## Core Concepts

### Data Model Hierarchy
- **Event** - A real-world question (e.g., "Will Bitcoin reach $100k?")
- **Market** - A tradeable outcome within an event, contains `clobTokenIds`
- **Token** - Each market has YES and NO token IDs used for CLOB trading
- **Prices** always 0.00 to 1.00 USDC. YES + NO prices sum to ~1.00

### Authentication Levels
- **L0 (Public)**: No auth. Read market data, prices, order books.
- **L1**: Private key required. Create/derive API credentials, sign orders locally.
- **L2**: API credentials required. Post orders, cancel orders, check balances.

### Signature Types (SignatureTypeV2)
| Type | Value | When to Use |
|------|-------|-------------|
| EOA | 0 | MetaMask, hardware wallets (needs POL for gas) |
| POLY_PROXY | 1 | Magic Link email/Google login (exported PK) |
| POLY_GNOSIS_SAFE | 2 | Gnosis Safe proxy wallet |
| POLY_1271 | 3 | EIP-1271 deposit wallets (new pUSD-era accounts) |

### Funder Address
The address that actually holds funds on Polymarket. For proxy wallets, the signing key differs from the funded address. Find it at polymarket.com/settings under "Wallet Address".

### APIs Overview
| API | Base URL | Purpose |
|-----|----------|---------|
| CLOB | `https://clob.polymarket.com` | Trading, order book, prices |
| Gamma | `https://gamma-api.polymarket.com` | Market discovery, events, metadata |
| Data API | Via CLOB endpoints | Positions, trades, activity |
| WebSocket | `wss://ws-subscriptions-clob.polymarket.com/ws/...` | Real-time order book & user updates |
| RTDS | Real-time data stream | Crypto prices (Binance/Chainlink), comments |

## Common Patterns

### Fetch Active Events (Gamma API)

```bash
curl "https://gamma-api.polymarket.com/events?active=true&closed=false&limit=5"
```

Response contains events with nested markets, each market has `clobTokenIds` array (index 0 = YES, index 1 = NO).

### Get Price & Order Book (CLOB API)

```bash
curl "https://clob.polymarket.com/price?token_id=TOKEN_ID&side=buy"
curl "https://clob.polymarket.com/book?token_id=TOKEN_ID"
```

### Place a Limit Order (GTC)

```python
from py_clob_client_v2.clob_types import OrderArgs, OrderType
from py_clob_client_v2.order_builder.constants import BUY

order = OrderArgs(token_id="<token-id>", price=0.50, size=10.0, side=BUY)
signed = client.create_order(order)
resp = client.post_order(signed, OrderType.GTC, post_only=False)
# Or one-shot: client.create_and_post_order(order, order_type=OrderType.GTC)
```

### Place a Market Order (FOK)

```python
from py_clob_client_v2.clob_types import MarketOrderArgs, OrderType
from py_clob_client_v2.order_builder.constants import BUY

mo = MarketOrderArgs(token_id="<token-id>", amount=25.0, side=BUY, order_type=OrderType.FOK)
signed = client.create_market_order(mo)
resp = client.post_order(signed, OrderType.FOK)
# Or one-shot: client.create_and_post_market_order(mo, order_type=OrderType.FOK)
```

### Cancel Orders

```python
from py_clob_client_v2.clob_types import OrderPayload

open_orders = client.get_open_orders()
if open_orders:
    client.cancel_order(OrderPayload(orderID=open_orders[0]["id"]))  # cancel one
client.cancel_all()                                                  # cancel all
```

### Batch Orders (up to 15)

Use the batch orders endpoint to submit multiple trades in a single request.

### Discover Token IDs from Gamma API

```python
import requests

events = requests.get(
    "https://gamma-api.polymarket.com/events",
    params={"active": "true", "closed": "false", "limit": 10}
).json()

for event in events:
    for market in event.get("markets", []):
        token_ids = market.get("clobTokenIds", [])
        # token_ids[0] = YES token, token_ids[1] = NO token
```

### Sports Markets

```bash
# List sports leagues
curl "https://gamma-api.polymarket.com/sports"
# Events by league
curl "https://gamma-api.polymarket.com/events?series_id=10345&active=true&closed=false"
```

## API Reference

### CLOB Client Key Methods (Python)

**Public (L0):**
- `get_ok()` - Health check
- `get_server_time()` - Server time
- `get_simplified_markets()` - List markets
- `get_midpoint(token_id)` - Mid price
- `get_price(token_id, side)` - Best price for side
- `get_order_book(token_id)` - Full order book
- `get_order_books([BookParams(...)])` - Multiple books
- `get_last_trade_price(token_id)` - Last trade price

**L1 (Private key):**
- `create_or_derive_api_key()` - Get or create API credentials
- `create_order(OrderArgs)` - Sign a limit order locally
- `create_market_order(MarketOrderArgs)` - Sign a market order locally

**L2 (API creds):**
- `post_order(signed_order, order_type, post_only=False)` - Submit signed order
- `post_orders(args, post_only=False)` - Batch submit
- `create_and_post_order(...)` / `create_and_post_market_order(...)` - One-shot helpers
- `get_open_orders(params=None)` - Get open orders
- `get_trades()` / `get_trades_paginated()` - Trade history
- `get_order(order_id)` - Single order status
- `cancel_order(OrderPayload(orderID=...))` - Cancel specific order
- `cancel_orders([hashes])` / `cancel_all()` / `cancel_market_orders(...)` - Bulk cancel
- `get_balance_allowance(BalanceAllowanceParams)` - pUSD collateral balance
- `get_pre_migration_orders()` - Orders from before the V2 cutover

### Order Types
| Type | Behavior |
|------|----------|
| GTC | Good 'til cancelled - stays on book |
| FOK | Fill or Kill - entire order fills instantly or cancelled |
| FAK | Fill and Kill - partial fill allowed, remainder cancelled |
| GTD | Good 'til date - expires at specified time |

### REST API Auth Headers (L1)
| Header | Description |
|--------|-------------|
| `POLY_ADDRESS` | Polygon signer address |
| `POLY_SIGNATURE` | EIP-712 signature |
| `POLY_TIMESTAMP` | UNIX timestamp |
| `POLY_NONCE` | Nonce (default 0) |

### REST API Auth Headers (L2)
| Header | Description |
|--------|-------------|
| `POLY_ADDRESS` | Polygon signer address |
| `POLY_SIGNATURE` | HMAC signature |
| `POLY_TIMESTAMP` | UNIX timestamp |
| `POLY_API_KEY` | API key |
| `POLY_PASSPHRASE` | API passphrase |

### Rate Limits (as of June 1, 2026)
- POST /order and DELETE /order: 120,000 per 10 minutes (200/s sustained)
- /books: 500/10s; /book, /price: 1,500/10s
- GET /markets/keyset: max `limit` 100 (paginate for more)

### Key Contract Addresses (Polygon)
V2 deployed new Exchange contracts at the April 2026 cutover (CTF Exchange V2 + Neg Risk CTF Exchange V2) and switched collateral to **pUSD**. The pre-V2 addresses below are LEGACY — fetch current ones from docs.polymarket.com before any on-chain work:
- **USDC.e (legacy collateral)**: `0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174`
- **Conditional Tokens (CTF)**: `0x4D97DCd97eC945f40cF65F87097ACe5EA0476045`
- **V1 Exchange (legacy)**: `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E`
- **V1 Neg Risk Exchange (legacy)**: `0xC5d563A36AE78145C45a50134d48A1215220f80a`
- **Neg Risk Adapter**: `0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296`

### Token Allowances (EOA/MetaMask only)
EOA wallets must approve pUSD and CTF tokens for the V2 exchange contracts before trading. Email/Magic and deposit wallets have auto-set allowances.

### Conditional Token Framework (CTF)
- **Split**: Lock USDC to receive YES + NO tokens
- **Merge**: Combine YES + NO tokens back into USDC
- **Redeem**: After resolution, redeem winning tokens for USDC

### Negative Risk Markets
Markets with >2 outcomes use the neg risk adapter contract. The `neg_risk` flag on market data indicates this. Use `tick_size` from the order book metadata.

## Gotchas

- **V1 SDK is dead**: `py-clob-client` (≤0.34.6) signs V1 orders that the exchange rejects since April 28, 2026. If orders suddenly stopped working around that date, this is why. Migrate to `py-clob-client-v2`.
- **Signature type matters**: Using wrong `signature_type` causes `INVALID_SIGNATURE`. EOA=0, Magic/email=1, Gnosis=2, deposit wallet (EIP-1271)=3.
- **Funder address required for proxy wallets**: Without it, orders are rejected with "Invalid Funder Address".
- **Token allowances for EOA**: MetaMask/hardware wallet users MUST approve pUSD and CTF tokens for the V2 exchange contracts before trading. Only needs to be done once per wallet.
- **Prices are 0-1 pUSD**: Not percentages. A price of 0.65 means $0.65 per share.
- **Token IDs != Market IDs**: Use Gamma API to get `clobTokenIds` from markets. Index 0 = YES, Index 1 = NO.
- **API creds are per-nonce**: If you lose creds but have the nonce, use `derive_api_key()`. If you lose both, create fresh creds.
- **Order book `min_order_size`**: Check this field - orders below minimum will be rejected.
- **`tick_size`**: Orders must conform to tick size (e.g., "0.01" or "0.001"). Available in book metadata.
- **Rate limits**: POST/DELETE /order allow 120,000 per 10 min (200/s sustained) since June 2026.
- **Fees set at match time**: V2 removed `feeRateBps` from orders. Use the market's `feeSchedule` object for fee calculations (Fee Structure V2, March 2026). 15-min crypto markets still carry taker fees.
- **Batch orders max 15**: Batch endpoint accepts up to 15 orders per request.
- **WebSocket subscriptions**: No limit on market channel token subscriptions. Use `initial_dump: true` (default) to get initial book state. WS message formats were NOT changed by the V2 migration.
- **Gamma pagination**: Use keyset pagination (`GET /markets/keyset`, `GET /events/keyset`, max limit 100) instead of offset-based (April–May 2026). `closed` on `GET /markets` defaults to false.
- **Geographic restrictions**: Some regions are restricted. Check `geoblock` docs.
- **Chain ID is 137**: Polygon mainnet. V2 also supports Amoy testnet (80002).

## Sources

- https://docs.polymarket.com/developers/CLOB/quickstart (scraped: 2026-06-12, CLOB V2)
- https://docs.polymarket.com/changelog/changelog (scraped: 2026-06-12)
- https://pypi.org/project/py-clob-client-v2/ (scraped: 2026-06-12, v1.0.1)
- py_clob_client_v2 1.0.1 source introspection (2026-06-12)
- https://docs.polymarket.com/developers/CLOB/introduction (scraped: 2026-02-06)
- https://docs.polymarket.com/developers/CLOB/authentication (scraped: 2026-02-06)
- https://github.com/Polymarket/py-clob-client (V1, archived 2026-05-25)
