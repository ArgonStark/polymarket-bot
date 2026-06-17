---
name: polymarket-api-endpoints
description: Complete reference for all Polymarket API endpoints - CLOB REST API, Gamma API, Data API, Bridge API, WebSocket channels, RTDS. Covers every endpoint path, HTTP method, parameters, rate limits, auth requirements. Use when building against Polymarket APIs, making HTTP requests, debugging API calls, or checking rate limits.
---

# Polymarket API Endpoints

Complete endpoint reference for all Polymarket APIs. Use this when making HTTP requests, building integrations, or debugging API calls.

## Base URLs

| API | Base URL | Purpose |
|-----|----------|---------|
| CLOB | `https://clob.polymarket.com` | Trading, order book, prices |
| Gamma | `https://gamma-api.polymarket.com` | Market discovery, events, metadata |
| Data | `https://data-api.polymarket.com` | Positions, activity, trade history |
| Bridge | `https://bridge.polymarket.com` | Deposits & withdrawals |
| CLOB WSS | `wss://ws-subscriptions-clob.polymarket.com/ws/` | Real-time order book & user events |
| RTDS WSS | `wss://ws-live-data.polymarket.com` | Crypto prices, comments stream |

## CLOB API Endpoints

### Health / Status (Public)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check ("ok") |
| `GET` | `/server-time` | Server time |

### Market Data (Public, L0)

| Method | Path | Query Params | Description |
|--------|------|-------------|-------------|
| `GET` | `/price` | `token_id`, `side` (buy/sell) | Current price for a token |
| `GET` | `/prices` | `token_ids` (comma-separated), `side` | Prices for multiple tokens |
| `GET` | `/midpoint` | `token_id` | Midpoint price |
| `GET` | `/midpoints` | `token_ids` | Midpoints for multiple tokens |
| `GET` | `/book` | `token_id` | Order book (bids/asks). Returns `min_order_size`, `neg_risk`, `tick_size` |
| `GET` | `/books` | `token_ids` | Multiple order books. Same metadata fields |
| `GET` | `/last-trade-price` | `token_id` | Last trade price for a token |
| `GET` | `/markets/{condition_id}` | - | Market info by condition ID |
| `GET` | `/simplified-markets` | `next_cursor` | Paginated list of simplified markets |
| `GET` | `/tick-size` | `token_id` | Tick size for a market (`0.1`, `0.01`, `0.001`, `0.0001`) |
| `GET` | `/neg-risk` | `token_id` | Whether market uses neg risk adapter |
| `GET` | `/price-history` | `token_id`, `interval`, `fidelity`, `startTs`, `endTs` | Historical price timeseries |
| `GET` | `/fee-rate` | `token_id` | Fee rate for a market (bps). Returns `0` for fee-free markets |

### Authentication (L1 - Private Key)

| Method | Path | Auth Headers | Description |
|--------|------|-------------|-------------|
| `POST` | `/auth/api-key` | POLY_ADDRESS, POLY_SIGNATURE, POLY_TIMESTAMP, POLY_NONCE | Create API credentials |
| `GET` | `/auth/derive-api-key` | POLY_ADDRESS, POLY_SIGNATURE, POLY_TIMESTAMP, POLY_NONCE | Derive existing API credentials |
| `GET` | `/auth/api-keys` | L2 headers | List API keys |
| `DELETE` | `/auth/api-key` | L2 headers | Delete an API key |

### Order Management (L2 - API Creds)

| Method | Path | Body/Params | Description |
|--------|------|------------|-------------|
| `POST` | `/order` | `{order, orderType, owner}` | Place a single order. Supports `post_only` flag |
| `POST` | `/orders` | Array of order objects (max 15) | Batch place orders |
| `DELETE` | `/order/{orderID}` | - | Cancel a single order |
| `DELETE` | `/orders` | `{orderIDs: [...]}` | Cancel multiple orders |
| `DELETE` | `/cancel-all` | - | Cancel all open orders |
| `DELETE` | `/cancel-market-orders` | `{market}` (condition ID) | Cancel all orders for a market |

### Order/Trade Queries (L2 - API Creds)

| Method | Path | Query Params | Description |
|--------|------|-------------|-------------|
| `GET` | `/order` | `id` | Get a specific order |
| `GET` | `/orders` | `market`, `asset_id` | Get open orders (optionally filtered) |
| `GET` | `/trades` | `market`, `asset_id` | Get user trade history |
| `GET` | `/notifications` | - | Get user notifications/fills |
| `GET` | `/data/orders` | various filters | Historical order data |
| `GET` | `/data/trades` | various filters, `limit` (max 500), `offset` (max 1000) | Historical trade data |

### Balance & Allowance (L2 - API Creds)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/balance-allowance` | Get balance and allowance info |
| `PUT` | `/balance-allowance` | Update balance allowance |

### Heartbeat (L2 - API Creds)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/heartbeat` | Connection monitoring, order cancellation on disconnect |

## Gamma API Endpoints

All Gamma endpoints are **public** (no auth required).

### Discovery & Metadata

| Method | Path | Query Params | Description |
|--------|------|-------------|-------------|
| `GET` | `/events` | `active`, `closed`, `limit`, `offset`, `order`, `ascending`, `tag_id`, `series_id`, `slug` | List events with filters |
| `GET` | `/events/{id}` | - | Get event by ID |
| `GET` | `/markets` | `slug`, `limit`, `offset`, `active`, `closed`, `clob_token_ids` | List/filter markets |
| `GET` | `/markets/{id}` | - | Get market by ID |
| `GET` | `/sports` | - | List all supported sports leagues |
| `GET` | `/tags` | `limit` | Get all available tags/topics |
| `GET` | `/series` | various filters | Get event series |
| `GET` | `/comments` | `market_id`, `event_id` | Get comments |
| `GET` | `/profiles` | query filters | Get user profiles |
| `GET` | `/search` | `query` | Search events and markets |

### Key Gamma Query Patterns

```bash
# Active events
curl "https://gamma-api.polymarket.com/events?active=true&closed=false&limit=10"

# Events by sports league
curl "https://gamma-api.polymarket.com/events?series_id=10345&active=true&closed=false"

# Sports game bets only (not futures)
curl "https://gamma-api.polymarket.com/events?series_id=10345&tag_id=100639&active=true&closed=false&order=startTime&ascending=true"

# Events by topic tag
curl "https://gamma-api.polymarket.com/events?tag_id=2&active=true&closed=false"

# Market by slug
curl "https://gamma-api.polymarket.com/markets?slug=will-bitcoin-reach-100k-by-2025"

# All tags
curl "https://gamma-api.polymarket.com/tags?limit=100"
```

### Gamma Response: Event with Markets

```json
{
  "id": "123456",
  "slug": "will-bitcoin-reach-100k-by-2025",
  "title": "Will Bitcoin reach $100k by 2025?",
  "active": true,
  "closed": false,
  "tags": [{"id": "21", "label": "Crypto", "slug": "crypto"}],
  "markets": [{
    "id": "789",
    "question": "Will Bitcoin reach $100k by 2025?",
    "clobTokenIds": ["TOKEN_YES_ID", "TOKEN_NO_ID"],
    "outcomes": "[\"Yes\", \"No\"]",
    "outcomePrices": "[\"0.65\", \"0.35\"]"
  }]
}
```

## Data API Endpoints

Base: `https://data-api.polymarket.com`

| Method | Path | Query Params | Description |
|--------|------|-------------|-------------|
| `GET` | `/positions` | `user` (address) | User's open positions |
| `GET` | `/closed-positions` | `user` (address) | User's closed positions |
| `GET` | `/activity` | `user`, `limit` (max 500), `offset` (max 1000) | User activity |
| `GET` | `/trades` | `user`, `limit` (max 500), `offset` (max 1000) | User trade history |

## Bridge API Endpoints

Base: `https://bridge.polymarket.com`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/supported-assets` | Get all supported chains and tokens |
| `POST` | `/quote` | Get a quote for deposit or withdrawal |
| `POST` | `/deposit` | Create deposit addresses for bridging assets in |
| `POST` | `/withdraw` | Withdraw USDC.e to any supported chain (EVM, Solana, Bitcoin) |
| `GET` | `/status/{address}` | Transaction status for a given address |

## Relayer API

| Method | Path | Description | Rate Limit |
|--------|------|-------------|------------|
| `POST` | `/submit` | Submit orders via Builder Relayer | 25 req/min |

## WebSocket Channels

### CLOB WebSocket (`wss://ws-subscriptions-clob.polymarket.com/ws/`)

**Market Channel** (public, subscribe with `assets_ids`):

| Event Type | Trigger | Key Fields |
|-----------|---------|------------|
| `book` | On subscribe + after trades | `asset_id`, `bids[]`, `asks[]`, `timestamp`, `hash` |
| `price_change` | Order placed/cancelled | `price_changes[]{asset_id, price, size, side, best_bid, best_ask}` |
| `tick_size_change` | Price reaches <0.04 or >0.96 | `old_tick_size`, `new_tick_size` |
| `last_trade_price` | Trade executed | `price`, `size`, `side`, `fee_rate_bps` |
| `best_bid_ask` | Best prices change | `best_bid`, `best_ask`, `spread` |
| `new_market` | Market created | `id`, `question`, `assets_ids`, `outcomes`, `event_message` |
| `market_resolved` | Market resolved | `winning_asset_id`, `winning_outcome` |

**User Channel** (authenticated, subscribe with `markets` condition IDs):

| Event Type | Trigger | Key Fields |
|-----------|---------|------------|
| `trade` | Order matched | `id`, `price`, `size`, `side`, `status`, `maker_orders[]` |
| `order` | Order placed/updated/cancelled | `id`, `price`, `side`, `type` (PLACEMENT/UPDATE/CANCELLATION) |

**Trade Statuses**: MATCHED -> MINED -> CONFIRMED (or RETRYING -> FAILED)

**Subscription Message Format**:
```json
{
  "auth": {},
  "markets": ["condition_id_1"],
  "assets_ids": ["token_id_1", "token_id_2"],
  "type": "market",
  "custom_feature_enabled": false
}
```

**Dynamic Subscribe/Unsubscribe** (after initial connection):
```json
{
  "assets_ids": ["new_token_id"],
  "operation": "subscribe"
}
```

### RTDS WebSocket (`wss://ws-live-data.polymarket.com`)

| Channel | Description |
|---------|-------------|
| Crypto Prices | Real-time prices from Binance & Chainlink |
| Comments | Real-time comment events (new, replies, reactions) |

## Rate Limits

### CLOB API

| Endpoint | Burst (per 10s) | Sustained (per 10min) |
|----------|----------------|----------------------|
| `POST /order` | 3,500 (500/s) | 120,000 (200/s, raised 2026-06-01) |
| `DELETE /order` | 3,000 (300/s) | 120,000 (200/s, raised 2026-06-01) |
| `POST /orders` (batch) | 1,000 (100/s) | 15,000 (25/s) |
| `DELETE /orders` | 1,000 (100/s) | 15,000 (25/s) |
| `DELETE /cancel-all` | 250 (25/s) | 6,000 (10/s) |
| `DELETE /cancel-market-orders` | 1,000 (100/s) | 1,500 (25/s) |
| `GET /book` | 1,500 | - |
| `GET /books` | 500 | - |
| `GET /price` | 1,500 | - |
| `GET /prices` | 500 | - |
| `GET /midprice` | 1,500 | - |
| `GET /midprices` | 500 | - |
| Ledger endpoints | 900 | - |
| `/data/orders` | 500 | - |
| `/data/trades` | 500 | - |
| `/notifications` | 125 | - |
| Price History | 1,000 | - |
| Tick Size | 200 | - |
| API Keys | 100 | - |
| Balance Allowance GET | 200 | - |
| Balance Allowance UPDATE | 50 | - |
| General CLOB | 9,000 | - |

### Gamma API

| Endpoint | Limit (per 10s) |
|----------|----------------|
| General | 4,000 |
| `/events` | 500 |
| `/markets` | 300 |
| `/markets` + `/events` listing | 900 |
| Comments | 200 |
| Tags | 200 |
| Search | 350 |

### Data API

| Endpoint | Limit (per 10s) |
|----------|----------------|
| General | 1,000 |
| `/trades` | 200 |
| `/positions` | 150 |
| `/closed-positions` | 150 |

### Other

| Endpoint | Limit |
|----------|-------|
| General global | 15,000 / 10s |
| "OK" endpoint | 100 / 10s |
| Relayer `/submit` | 25 / 1 min |
| User PNL | 200 / 10s |

## Auth Headers Reference

### L1 (Create/Derive API Keys)

| Header | Value |
|--------|-------|
| `POLY_ADDRESS` | Polygon signer address |
| `POLY_SIGNATURE` | EIP-712 signed message |
| `POLY_TIMESTAMP` | UNIX timestamp |
| `POLY_NONCE` | Nonce (default 0) |

### L2 (Trading Operations)

| Header | Value |
|--------|-------|
| `POLY_ADDRESS` | Polygon signer address |
| `POLY_SIGNATURE` | HMAC signature |
| `POLY_TIMESTAMP` | UNIX timestamp |
| `POLY_API_KEY` | API key from L1 auth |
| `POLY_PASSPHRASE` | Passphrase from L1 auth |

## Order Types

| Type | Behavior |
|------|----------|
| GTC | Good 'til cancelled - rests on book |
| GTD | Good 'til date - auto-expires at specified time |
| FOK | Fill or Kill - entire order fills instantly or cancelled |
| FAK | Fill and Kill - partial fill allowed, remainder cancelled |
| Post Only | Rejected if it would immediately match (maker-only) |

## Gotchas

- **Data API pagination limits**: `limit` max 500, `offset` max 1000 on `/trades` and `/activity`
- **Batch orders max 15**: `/orders` endpoint accepts up to 15 orders per request
- **Rate limits are throttled, not rejected**: Excess requests are delayed/queued, not dropped
- **WSS no subscription limit**: Market channel has no token subscription limit (changed May 2025)
- **`initial_dump` default true**: WSS sends initial book state on subscribe unless set to false
- **Tick size changes dynamically**: When price reaches <0.04 or >0.96, tick size changes (notified via WSS)
- **CLOB V2 (April 28, 2026)**: `feeRateBps`, `nonce`, `taker` were REMOVED from the order struct — fees are set at match time from the market's `feeSchedule`. Orders signed by V1 SDKs are rejected; use `py-clob-client-v2` / `@polymarket/clob-client-v2`. Collateral is pUSD (was USDC.e). WS message formats unchanged.
- **Keyset pagination**: `GET /markets/keyset` and `GET /events/keyset` (April 2026) replace offset pagination; max `limit` 100 since May 2026. `closed` on `GET /markets` defaults to false.
- **Relayer `POST /submit`** returns immediately without `transactionHash` (April 2026); poll `GET /transaction` for the hash
- **`/book` metadata**: The `/book` and `/books` endpoints return `min_order_size`, `neg_risk`, and `tick_size`

## Sources

- https://docs.polymarket.com/quickstart/reference/endpoints (scraped: 2026-02-06)
- https://docs.polymarket.com/quickstart/introduction/rate-limits (scraped: 2026-02-06)
- https://docs.polymarket.com/developers/gamma-markets-api/overview (scraped: 2026-02-06)
- https://docs.polymarket.com/developers/CLOB/rest-api/* (crawled: 2026-02-06)
- https://docs.polymarket.com/developers/gamma-markets-api/* (crawled: 2026-02-06)
- https://docs.polymarket.com/developers/CLOB/websocket/wss-overview (scraped: 2026-02-06)
- https://docs.polymarket.com/developers/CLOB/websocket/market-channel (scraped: 2026-02-06)
- https://docs.polymarket.com/developers/CLOB/websocket/user-channel (scraped: 2026-02-06)
- https://docs.polymarket.com/developers/misc-endpoints/bridge-overview (scraped: 2026-02-06)
