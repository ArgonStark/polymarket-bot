"""
Probability calculation module for temporal arbitrage.

Uses log-normal price dynamics to calculate true probabilities
of price outcomes based on Chainlink oracle prices.
"""

import math
from typing import Optional
import numpy as np
from scipy.stats import norm


# Duration of 15-minute market in seconds
MARKET_DURATION_SECONDS = 900.0


def calculate_true_probability(
    current_price: float,
    target_price: float,
    time_remaining_sec: float,
    volatility_15min: float,
) -> float:
    """
    Calculate probability that price stays above target at settlement.

    Uses log-normal price dynamics to estimate the probability that
    the end price will be >= target_price given current conditions.

    For UP market: P(end_price >= target)
    For DOWN market: P(end_price < target) = 1 - P(end_price >= target)

    Args:
        current_price: Current Chainlink price
        target_price: Start/reference price for the market
        time_remaining_sec: Seconds until settlement
        volatility_15min: Historical 15-minute volatility (e.g., 0.003 for BTC)

    Returns:
        Probability that price will be >= target at settlement (0.0 to 1.0)
    """
    # Edge cases
    if time_remaining_sec <= 0:
        # Market has expired, outcome is deterministic
        return 1.0 if current_price >= target_price else 0.0

    if current_price <= 0 or target_price <= 0:
        return 0.5  # Invalid prices, return neutral

    if volatility_15min <= 0:
        # No volatility means price won't move
        return 1.0 if current_price >= target_price else 0.0

    # Scale volatility to remaining time using square root of time
    time_fraction = time_remaining_sec / MARKET_DURATION_SECONDS
    scaled_vol = volatility_15min * math.sqrt(time_fraction)

    if scaled_vol <= 0:
        return 1.0 if current_price >= target_price else 0.0

    # Log return from current to target
    # Positive means current > target (UP is likely)
    # Negative means current < target (DOWN is likely)
    log_distance = math.log(current_price / target_price)

    # Z-score: how many standard deviations is current price from target?
    # High positive z-score means current price is well above target
    z_score = log_distance / scaled_vol

    # Probability price stays above target
    # Using CDF: P(X <= z) where X ~ N(0,1)
    # We want P(end_price >= target) = P(log_return >= log(target/current))
    # This equals P(Z >= -log_distance/vol) = P(Z <= log_distance/vol) = CDF(z_score)
    prob_up = norm.cdf(z_score)

    # Clamp to valid probability range
    return max(0.0001, min(0.9999, prob_up))


def calculate_edge(
    true_prob: float,
    market_prob: float,
) -> float:
    """
    Calculate edge over market odds.

    Args:
        true_prob: Calculated true probability of outcome
        market_prob: Market's implied probability (cost to buy)

    Returns:
        Edge as decimal (e.g., 0.25 for 25% edge)
    """
    return true_prob - market_prob


def estimate_volatility(
    price_history: list[float],
    window: int = 20,
    default_vol: float = 0.003,
) -> float:
    """
    Estimate 15-minute volatility from recent price data.

    Uses realized volatility from log returns of historical prices.

    Args:
        price_history: List of prices at 15-min intervals (most recent last)
        window: Number of periods to use for calculation
        default_vol: Default volatility if insufficient data

    Returns:
        Estimated 15-minute volatility as decimal
    """
    if len(price_history) < 2:
        return default_vol

    # Use only the most recent window prices
    prices = price_history[-min(len(price_history), window + 1):]

    if len(prices) < 2:
        return default_vol

    # Calculate log returns
    log_returns = []
    for i in range(1, len(prices)):
        if prices[i - 1] > 0 and prices[i] > 0:
            log_returns.append(math.log(prices[i] / prices[i - 1]))

    if not log_returns:
        return default_vol

    # Standard deviation of log returns
    vol = float(np.std(log_returns))

    # Apply minimum volatility floor
    return max(vol, 0.0001)


def estimate_vol_15m_from_ticks(
    history: list,
    default_vol: float = 0.005,
    min_span_seconds: float = 120.0,
    max_gap_seconds: float = 120.0,
    max_ratio: float = 4.0,
) -> float:
    """
    Time-aware realized volatility from irregular (timestamp, price) ticks,
    correctly scaled to a 15-minute horizon.

    The legacy estimate_volatility() treats tick-to-tick returns (arriving
    every few seconds) as if they were 15-minute returns — understating
    volatility by ~sqrt(900/dt), i.e. 15-30x. This version estimates the
    per-second realized variance rate and scales by sqrt(900):

        var_rate = sum(r_i^2) / sum(dt_i)      (per-second variance)
        sigma_15m = sqrt(var_rate * 900)

    Args:
        history: List of (datetime, price) tuples, oldest first.
        default_vol: Fallback / anchor when data is insufficient.
        min_span_seconds: Minimum total time span required to trust the estimate.
        max_gap_seconds: Ignore intervals longer than this (feed reconnects).
        max_ratio: Clamp result to [default/max_ratio, default*max_ratio] —
                   resists quiet-feed underestimates and single-tick spikes.

    Returns:
        Estimated 15-minute volatility as a decimal fraction.
    """
    if not history or len(history) < 5:
        return default_vol

    sum_r2 = 0.0
    sum_dt = 0.0
    prev_t = None
    prev_p = None
    for item in history:
        if not (isinstance(item, (list, tuple)) and len(item) >= 2):
            continue
        t, p = item[0], item[1]
        if prev_p is not None and p > 0 and prev_p > 0:
            try:
                dt = (t - prev_t).total_seconds()
            except (TypeError, AttributeError):
                dt = 0.0
            if 0.0 < dt <= max_gap_seconds:
                r = math.log(p / prev_p)
                sum_r2 += r * r
                sum_dt += dt
        prev_t, prev_p = t, p

    if sum_dt < min_span_seconds or sum_r2 <= 0.0:
        return default_vol

    sigma_15m = math.sqrt(sum_r2 / sum_dt * 900.0)

    lo = default_vol / max_ratio
    hi = default_vol * max_ratio
    return min(hi, max(lo, sigma_15m))


def estimate_volatility_from_range(
    high: float,
    low: float,
    duration_minutes: float = 15.0,
) -> float:
    """
    Estimate volatility from high-low range (Parkinson estimator).

    Useful when only range data is available.

    Args:
        high: Highest price in period
        low: Lowest price in period
        duration_minutes: Duration of the range period

    Returns:
        Estimated volatility scaled to 15 minutes
    """
    if high <= 0 or low <= 0 or high <= low:
        return 0.003  # Default

    # Parkinson volatility estimator
    # sigma = (1 / (2 * sqrt(ln(2)))) * ln(high/low)
    parkinson_factor = 1.0 / (2.0 * math.sqrt(math.log(2)))
    vol = parkinson_factor * math.log(high / low)

    # Scale to 15 minutes if needed
    if duration_minutes != 15.0:
        vol *= math.sqrt(15.0 / duration_minutes)

    return max(vol, 0.0001)


def calculate_taker_fee(price: float, shares: float = 100.0) -> float:
    """
    Calculate taker fee for a trade.

    Fee formula: 0.25 * (price * (1 - price))^2

    Args:
        price: Trade price (0.01 - 0.99)
        shares: Number of shares traded

    Returns:
        Fee amount in USD
    """
    # Clamp price to valid range
    price = max(0.01, min(0.99, price))

    # Fee rate
    fee_rate = 0.25 * (price * (1 - price)) ** 2

    # Total fee
    return fee_rate * shares * price


def calculate_breakeven_edge(price: float) -> float:
    """
    Calculate minimum edge needed to break even after taker fees.

    Args:
        price: Expected trade price

    Returns:
        Minimum edge required (as decimal)
    """
    price = max(0.01, min(0.99, price))
    fee_rate = 0.25 * (price * (1 - price)) ** 2
    return fee_rate


def optimal_kelly_fraction(edge: float, odds: float) -> float:
    """
    Calculate Kelly criterion for optimal position sizing.

    Kelly fraction = (p * b - q) / b
    where p = probability of win, b = odds, q = probability of loss

    Args:
        edge: Edge over market (e.g., 0.25 for 25%)
        odds: Implied odds from market (cost to buy)

    Returns:
        Optimal fraction of bankroll to bet
    """
    if odds <= 0 or odds >= 1:
        return 0.0

    # True probability = market probability + edge
    true_prob = odds + edge
    true_prob = max(0.01, min(0.99, true_prob))

    # Odds multiplier (payout on $1 bet if win)
    # If you pay 'odds' and win, you get $1, so profit is (1 - odds)
    # Odds ratio b = profit / cost = (1 - odds) / odds
    b = (1 - odds) / odds

    # Kelly formula
    p = true_prob
    q = 1 - p
    kelly = (p * b - q) / b if b > 0 else 0.0

    # Apply fractional Kelly (half Kelly is common for safety)
    # and cap at reasonable maximum
    kelly = max(0.0, min(kelly * 0.5, 0.25))

    return kelly


def time_decay_factor(time_remaining_sec: float) -> float:
    """
    Calculate time decay factor for position sizing.

    Less time = more certain outcome = potentially larger position.

    Args:
        time_remaining_sec: Seconds until settlement

    Returns:
        Multiplier for position size (0.5 to 2.0)
    """
    # At 15 minutes: factor = 1.0
    # At 7.5 minutes: factor ~= 1.4
    # At 1 minute: factor ~= 2.0
    # At 0 seconds: factor = 0 (no trading)

    if time_remaining_sec <= 0:
        return 0.0

    if time_remaining_sec >= MARKET_DURATION_SECONDS:
        return 1.0

    # Square root scaling - positions grow as certainty increases
    time_fraction = time_remaining_sec / MARKET_DURATION_SECONDS
    factor = 1.0 / math.sqrt(time_fraction)

    # Cap the factor
    return min(factor, 2.0)


def confidence_adjusted_probability(
    true_prob: float,
    time_remaining_sec: float,
    volatility: float,
) -> tuple[float, float]:
    """
    Calculate confidence interval for true probability.

    As time decreases and volatility is known, confidence increases.

    Args:
        true_prob: Calculated true probability
        time_remaining_sec: Seconds until settlement
        volatility: Estimated volatility

    Returns:
        Tuple of (lower_bound, upper_bound) for 95% confidence
    """
    # Uncertainty decreases with less time
    time_fraction = max(0.01, time_remaining_sec / MARKET_DURATION_SECONDS)

    # Base uncertainty from volatility estimation error (~20%)
    base_uncertainty = 0.20 * volatility * 10  # Scale to probability space

    # Time-scaled uncertainty
    uncertainty = base_uncertainty * math.sqrt(time_fraction)

    # Probability space transformation
    prob_uncertainty = uncertainty * true_prob * (1 - true_prob) * 4

    lower = max(0.0, true_prob - prob_uncertainty)
    upper = min(1.0, true_prob + prob_uncertainty)

    return (lower, upper)
