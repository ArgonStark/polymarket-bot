#!/usr/bin/env python3
"""
Performance Dashboard - Analyze Enriched Historical Trades

Provides insights into:
- Best/worst trading conditions
- Time-based patterns (hour, day, session)
- Technical indicator effectiveness
- Asset performance breakdown
- Feature correlation with outcomes

Usage:
    python scripts/performance_dashboard.py
    python scripts/performance_dashboard.py --file data/enriched_trades.json
"""

import sys
import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_enriched_trades(file_path: str) -> List[Dict]:
    """Load enriched trades from JSON file."""
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {file_path}")
        return []

    with open(path, 'r') as f:
        data = json.load(f)

    # Handle both list and dict formats
    if isinstance(data, dict):
        return data.get('trades', [])
    return data


def print_header(title: str, char: str = "="):
    """Print a formatted section header."""
    print(f"\n{char * 60}")
    print(f"  {title}")
    print(f"{char * 60}")


def print_subheader(title: str):
    """Print a formatted subheader."""
    print(f"\n  {title}")
    print(f"  {'-' * len(title)}")


def calculate_win_rate(trades: List[Dict]) -> Tuple[int, int, float]:
    """Calculate win rate from trades."""
    if not trades:
        return 0, 0, 0.0
    wins = sum(1 for t in trades if t.get('outcome') == 'WIN')
    total = len(trades)
    return wins, total, wins / total if total > 0 else 0.0


def format_pnl(pnl: float) -> str:
    """Format P&L with color indicator."""
    if pnl >= 0:
        return f"+${pnl:,.2f}"
    return f"-${abs(pnl):,.2f}"


def analyze_overview(trades: List[Dict]) -> Dict:
    """Get overall performance overview."""
    wins, total, win_rate = calculate_win_rate(trades)

    total_pnl = sum(t.get('realized_pnl', 0) for t in trades)
    total_size = sum(t.get('size_usd', 0) for t in trades)

    winning_trades = [t for t in trades if t.get('outcome') == 'WIN']
    losing_trades = [t for t in trades if t.get('outcome') == 'LOSS']

    avg_win = sum(t.get('realized_pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(t.get('realized_pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0

    return {
        'total_trades': total,
        'wins': wins,
        'losses': total - wins,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'total_volume': total_size,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': abs(avg_win * wins / (avg_loss * (total - wins))) if avg_loss != 0 and (total - wins) > 0 else float('inf'),
    }


def analyze_by_asset(trades: List[Dict]) -> Dict[str, Dict]:
    """Analyze performance by asset."""
    by_asset = defaultdict(list)
    for t in trades:
        asset = t.get('asset', 'Unknown')
        by_asset[asset].append(t)

    results = {}
    for asset, asset_trades in by_asset.items():
        wins, total, win_rate = calculate_win_rate(asset_trades)
        pnl = sum(t.get('realized_pnl', 0) for t in asset_trades)
        results[asset] = {
            'trades': total,
            'wins': wins,
            'win_rate': win_rate,
            'pnl': pnl,
            'avg_pnl': pnl / total if total > 0 else 0,
        }

    return dict(sorted(results.items(), key=lambda x: x[1]['pnl'], reverse=True))


def analyze_by_side(trades: List[Dict]) -> Dict[str, Dict]:
    """Analyze performance by trade side (UP/DOWN)."""
    by_side = defaultdict(list)
    for t in trades:
        side = t.get('side', 'Unknown')
        by_side[side].append(t)

    results = {}
    for side, side_trades in by_side.items():
        wins, total, win_rate = calculate_win_rate(side_trades)
        pnl = sum(t.get('realized_pnl', 0) for t in side_trades)
        results[side] = {
            'trades': total,
            'wins': wins,
            'win_rate': win_rate,
            'pnl': pnl,
        }

    return results


def analyze_by_hour(trades: List[Dict]) -> Dict[int, Dict]:
    """Analyze performance by hour of day."""
    by_hour = defaultdict(list)
    for t in trades:
        hour = t.get('hour_utc', 0)
        by_hour[hour].append(t)

    results = {}
    for hour in range(24):
        hour_trades = by_hour.get(hour, [])
        if hour_trades:
            wins, total, win_rate = calculate_win_rate(hour_trades)
            pnl = sum(t.get('realized_pnl', 0) for t in hour_trades)
            results[hour] = {
                'trades': total,
                'win_rate': win_rate,
                'pnl': pnl,
            }

    return results


def analyze_by_day(trades: List[Dict]) -> Dict[str, Dict]:
    """Analyze performance by day of week."""
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    by_day = defaultdict(list)

    for t in trades:
        day_idx = t.get('day_of_week', 0)
        by_day[day_idx].append(t)

    results = {}
    for idx, day_name in enumerate(days):
        day_trades = by_day.get(idx, [])
        if day_trades:
            wins, total, win_rate = calculate_win_rate(day_trades)
            pnl = sum(t.get('realized_pnl', 0) for t in day_trades)
            results[day_name] = {
                'trades': total,
                'win_rate': win_rate,
                'pnl': pnl,
            }

    return results


def analyze_by_session(trades: List[Dict]) -> Dict[str, Dict]:
    """Analyze performance by trading session."""
    sessions = {
        'Asia': 'is_asia_session',
        'Europe': 'is_europe_session',
        'US': 'is_us_session',
    }

    results = {}
    for session_name, field in sessions.items():
        session_trades = [t for t in trades if t.get(field, False)]
        if session_trades:
            wins, total, win_rate = calculate_win_rate(session_trades)
            pnl = sum(t.get('realized_pnl', 0) for t in session_trades)
            results[session_name] = {
                'trades': total,
                'win_rate': win_rate,
                'pnl': pnl,
            }

    # Weekend
    weekend_trades = [t for t in trades if t.get('is_weekend', False)]
    if weekend_trades:
        wins, total, win_rate = calculate_win_rate(weekend_trades)
        pnl = sum(t.get('realized_pnl', 0) for t in weekend_trades)
        results['Weekend'] = {
            'trades': total,
            'win_rate': win_rate,
            'pnl': pnl,
        }

    return results


def analyze_by_trend(trades: List[Dict]) -> Dict[str, Dict]:
    """Analyze performance by trend direction."""
    by_trend = defaultdict(list)
    for t in trades:
        trend = t.get('trend_direction', 'neutral')
        by_trend[trend].append(t)

    results = {}
    for trend, trend_trades in by_trend.items():
        wins, total, win_rate = calculate_win_rate(trend_trades)
        pnl = sum(t.get('realized_pnl', 0) for t in trend_trades)
        results[trend.capitalize()] = {
            'trades': total,
            'win_rate': win_rate,
            'pnl': pnl,
        }

    return results


def analyze_by_rsi_zone(trades: List[Dict]) -> Dict[str, Dict]:
    """Analyze performance by RSI zone."""
    zones = {
        'Oversold (<30)': lambda t: t.get('rsi_14', 50) < 30,
        'Neutral (30-70)': lambda t: 30 <= t.get('rsi_14', 50) <= 70,
        'Overbought (>70)': lambda t: t.get('rsi_14', 50) > 70,
    }

    results = {}
    for zone_name, condition in zones.items():
        zone_trades = [t for t in trades if condition(t)]
        if zone_trades:
            wins, total, win_rate = calculate_win_rate(zone_trades)
            pnl = sum(t.get('realized_pnl', 0) for t in zone_trades)
            results[zone_name] = {
                'trades': total,
                'win_rate': win_rate,
                'pnl': pnl,
            }

    return results


def analyze_by_volatility(trades: List[Dict]) -> Dict[str, Dict]:
    """Analyze performance by volatility level."""
    # Calculate volatility percentiles
    volatilities = [t.get('volatility_1h', 0) for t in trades if t.get('volatility_1h', 0) > 0]
    if not volatilities:
        return {}

    sorted_vol = sorted(volatilities)
    low_threshold = sorted_vol[len(sorted_vol) // 3] if len(sorted_vol) > 3 else 0
    high_threshold = sorted_vol[2 * len(sorted_vol) // 3] if len(sorted_vol) > 3 else float('inf')

    zones = {
        'Low': lambda t: 0 < t.get('volatility_1h', 0) <= low_threshold,
        'Medium': lambda t: low_threshold < t.get('volatility_1h', 0) <= high_threshold,
        'High': lambda t: t.get('volatility_1h', 0) > high_threshold,
    }

    results = {}
    for zone_name, condition in zones.items():
        zone_trades = [t for t in trades if condition(t)]
        if zone_trades:
            wins, total, win_rate = calculate_win_rate(zone_trades)
            pnl = sum(t.get('realized_pnl', 0) for t in zone_trades)
            results[zone_name] = {
                'trades': total,
                'win_rate': win_rate,
                'pnl': pnl,
            }

    return results


def analyze_by_bb_position(trades: List[Dict]) -> Dict[str, Dict]:
    """Analyze performance by Bollinger Band position."""
    by_bb = defaultdict(list)
    for t in trades:
        bb_pos = t.get('bb_position', 'middle')
        by_bb[bb_pos].append(t)

    results = {}
    for pos, pos_trades in by_bb.items():
        wins, total, win_rate = calculate_win_rate(pos_trades)
        pnl = sum(t.get('realized_pnl', 0) for t in pos_trades)
        results[pos.capitalize()] = {
            'trades': total,
            'win_rate': win_rate,
            'pnl': pnl,
        }

    return results


def analyze_by_stoch_signal(trades: List[Dict]) -> Dict[str, Dict]:
    """Analyze performance by Stochastic signal."""
    by_stoch = defaultdict(list)
    for t in trades:
        signal = t.get('stoch_signal', 'neutral')
        by_stoch[signal].append(t)

    results = {}
    for signal, signal_trades in by_stoch.items():
        wins, total, win_rate = calculate_win_rate(signal_trades)
        pnl = sum(t.get('realized_pnl', 0) for t in signal_trades)
        results[signal.capitalize()] = {
            'trades': total,
            'win_rate': win_rate,
            'pnl': pnl,
        }

    return results


def analyze_by_macd_signal(trades: List[Dict]) -> Dict[str, Dict]:
    """Analyze performance by MACD signal."""
    by_macd = defaultdict(list)
    for t in trades:
        signal = t.get('macd_signal', 'none')
        by_macd[signal].append(t)

    results = {}
    for signal, signal_trades in by_macd.items():
        wins, total, win_rate = calculate_win_rate(signal_trades)
        pnl = sum(t.get('realized_pnl', 0) for t in signal_trades)
        results[signal.capitalize()] = {
            'trades': total,
            'win_rate': win_rate,
            'pnl': pnl,
        }

    return results


def find_best_worst_conditions(trades: List[Dict]) -> Dict:
    """Find the best and worst trading conditions."""
    # Group by multiple factors
    conditions = defaultdict(list)

    for t in trades:
        # Create condition key
        key_parts = []

        # Trend
        trend = t.get('trend_direction', 'neutral')
        key_parts.append(f"trend:{trend}")

        # RSI zone
        rsi = t.get('rsi_14', 50)
        if rsi < 30:
            key_parts.append("rsi:oversold")
        elif rsi > 70:
            key_parts.append("rsi:overbought")
        else:
            key_parts.append("rsi:neutral")

        # Session
        if t.get('is_asia_session'):
            key_parts.append("session:asia")
        elif t.get('is_europe_session'):
            key_parts.append("session:europe")
        elif t.get('is_us_session'):
            key_parts.append("session:us")

        key = " | ".join(key_parts)
        conditions[key].append(t)

    # Calculate stats for each condition
    condition_stats = []
    for key, cond_trades in conditions.items():
        if len(cond_trades) >= 5:  # Minimum sample size
            wins, total, win_rate = calculate_win_rate(cond_trades)
            pnl = sum(t.get('realized_pnl', 0) for t in cond_trades)
            condition_stats.append({
                'condition': key,
                'trades': total,
                'win_rate': win_rate,
                'pnl': pnl,
                'avg_pnl': pnl / total,
            })

    # Sort by win rate
    condition_stats.sort(key=lambda x: x['win_rate'], reverse=True)

    return {
        'best': condition_stats[:5] if condition_stats else [],
        'worst': condition_stats[-5:][::-1] if condition_stats else [],
    }


def print_table(data: Dict, columns: List[str], title: str = None):
    """Print data as a formatted table."""
    if title:
        print_subheader(title)

    if not data:
        print("    No data available")
        return

    # Calculate column widths
    col_widths = {col: len(col) for col in columns}
    for key, values in data.items():
        col_widths['Name'] = max(col_widths.get('Name', 4), len(str(key)))
        for col in columns:
            if col == 'Name':
                continue
            val = values.get(col.lower().replace(' ', '_'), values.get(col.lower(), ''))
            if isinstance(val, float):
                if 'rate' in col.lower():
                    val_str = f"{val:.1%}"
                elif 'pnl' in col.lower():
                    val_str = format_pnl(val)
                else:
                    val_str = f"{val:.2f}"
            else:
                val_str = str(val)
            col_widths[col] = max(col_widths.get(col, 0), len(val_str))

    # Print header
    header = "    "
    for col in columns:
        header += f"{col:<{col_widths[col] + 2}}"
    print(header)
    print("    " + "-" * (sum(col_widths.values()) + len(columns) * 2))

    # Print rows
    for key, values in data.items():
        row = f"    {str(key):<{col_widths['Name'] + 2}}"
        for col in columns[1:]:
            val = values.get(col.lower().replace(' ', '_'), values.get(col.lower(), ''))
            if isinstance(val, float):
                if 'rate' in col.lower():
                    val_str = f"{val:.1%}"
                elif 'pnl' in col.lower():
                    val_str = format_pnl(val)
                else:
                    val_str = f"{val:.2f}"
            else:
                val_str = str(val)
            row += f"{val_str:<{col_widths[col] + 2}}"
        print(row)


def main():
    """Main dashboard entry point."""
    parser = argparse.ArgumentParser(description='Performance Dashboard')
    parser.add_argument('--file', '-f', default='data/enriched_trades.json',
                        help='Path to enriched trades file')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("     POLYMARKET BOT - PERFORMANCE DASHBOARD")
    print("=" * 60)

    # Load trades
    trades = load_enriched_trades(args.file)
    if not trades:
        print(f"\nNo trades found in {args.file}")
        print("Run the enrichment tool first:")
        print("  python -m src.tools.enrich_historical_trades --wallet <address>")
        return

    print(f"\nLoaded {len(trades)} enriched trades")

    # Overview
    print_header("OVERALL PERFORMANCE")
    overview = analyze_overview(trades)
    print(f"""
    Total Trades:    {overview['total_trades']}
    Wins/Losses:     {overview['wins']} / {overview['losses']}
    Win Rate:        {overview['win_rate']:.1%}

    Total P&L:       {format_pnl(overview['total_pnl'])}
    Total Volume:    ${overview['total_volume']:,.2f}

    Avg Win:         {format_pnl(overview['avg_win'])}
    Avg Loss:        {format_pnl(overview['avg_loss'])}
    Profit Factor:   {overview['profit_factor']:.2f}
    """)

    # By Asset
    print_header("PERFORMANCE BY ASSET")
    asset_data = analyze_by_asset(trades)
    print_table(asset_data, ['Name', 'Trades', 'Wins', 'Win Rate', 'PnL'])

    # By Side
    print_header("PERFORMANCE BY SIDE (UP/DOWN)")
    side_data = analyze_by_side(trades)
    print_table(side_data, ['Name', 'Trades', 'Wins', 'Win Rate', 'PnL'])

    # By Trading Session
    print_header("PERFORMANCE BY SESSION")
    session_data = analyze_by_session(trades)
    print_table(session_data, ['Name', 'Trades', 'Win Rate', 'PnL'])

    # By Day of Week
    print_header("PERFORMANCE BY DAY")
    day_data = analyze_by_day(trades)
    print_table(day_data, ['Name', 'Trades', 'Win Rate', 'PnL'])

    # By Hour (top/bottom 5)
    print_header("PERFORMANCE BY HOUR (UTC)")
    hour_data = analyze_by_hour(trades)
    if hour_data:
        sorted_hours = sorted(hour_data.items(), key=lambda x: x[1]['win_rate'], reverse=True)

        print_subheader("Best Hours")
        for hour, stats in sorted_hours[:5]:
            print(f"    {hour:02d}:00 UTC  -  {stats['trades']} trades, {stats['win_rate']:.1%} win rate, {format_pnl(stats['pnl'])}")

        print_subheader("Worst Hours")
        for hour, stats in sorted_hours[-5:]:
            print(f"    {hour:02d}:00 UTC  -  {stats['trades']} trades, {stats['win_rate']:.1%} win rate, {format_pnl(stats['pnl'])}")

    # Technical Indicators
    print_header("TECHNICAL INDICATOR ANALYSIS")

    # Trend
    print_subheader("By Trend Direction")
    trend_data = analyze_by_trend(trades)
    print_table(trend_data, ['Name', 'Trades', 'Win Rate', 'PnL'])

    # RSI
    print_subheader("By RSI Zone")
    rsi_data = analyze_by_rsi_zone(trades)
    print_table(rsi_data, ['Name', 'Trades', 'Win Rate', 'PnL'])

    # Bollinger Bands
    print_subheader("By Bollinger Band Position")
    bb_data = analyze_by_bb_position(trades)
    print_table(bb_data, ['Name', 'Trades', 'Win Rate', 'PnL'])

    # Stochastic
    print_subheader("By Stochastic Signal")
    stoch_data = analyze_by_stoch_signal(trades)
    print_table(stoch_data, ['Name', 'Trades', 'Win Rate', 'PnL'])

    # MACD
    print_subheader("By MACD Signal")
    macd_data = analyze_by_macd_signal(trades)
    print_table(macd_data, ['Name', 'Trades', 'Win Rate', 'PnL'])

    # Volatility
    print_subheader("By Volatility Level")
    vol_data = analyze_by_volatility(trades)
    print_table(vol_data, ['Name', 'Trades', 'Win Rate', 'PnL'])

    # Best/Worst Conditions
    print_header("BEST & WORST CONDITIONS")
    conditions = find_best_worst_conditions(trades)

    if conditions['best']:
        print_subheader("Best Conditions (min 5 trades)")
        for c in conditions['best']:
            print(f"    {c['condition']}")
            print(f"      {c['trades']} trades | {c['win_rate']:.1%} win rate | {format_pnl(c['pnl'])} P&L")

    if conditions['worst']:
        print_subheader("Worst Conditions (min 5 trades)")
        for c in conditions['worst']:
            print(f"    {c['condition']}")
            print(f"      {c['trades']} trades | {c['win_rate']:.1%} win rate | {format_pnl(c['pnl'])} P&L")

    # Summary recommendations
    print_header("INSIGHTS & RECOMMENDATIONS")

    insights = []

    # Best asset
    if asset_data:
        best_asset = max(asset_data.items(), key=lambda x: x[1]['win_rate'])
        if best_asset[1]['trades'] >= 10:
            insights.append(f"Best performing asset: {best_asset[0]} ({best_asset[1]['win_rate']:.1%} win rate)")

    # Best session
    if session_data:
        best_session = max(session_data.items(), key=lambda x: x[1]['win_rate'])
        if best_session[1]['trades'] >= 10:
            insights.append(f"Best trading session: {best_session[0]} ({best_session[1]['win_rate']:.1%} win rate)")

    # Best day
    if day_data:
        best_day = max(day_data.items(), key=lambda x: x[1]['win_rate'])
        if best_day[1]['trades'] >= 10:
            insights.append(f"Best day: {best_day[0]} ({best_day[1]['win_rate']:.1%} win rate)")

    # Trend insight
    if trend_data:
        best_trend = max(trend_data.items(), key=lambda x: x[1]['win_rate'])
        insights.append(f"Best trend condition: {best_trend[0]} ({best_trend[1]['win_rate']:.1%} win rate)")

    # Side insight
    if side_data:
        up_rate = side_data.get('UP', {}).get('win_rate', 0)
        down_rate = side_data.get('DOWN', {}).get('win_rate', 0)
        if abs(up_rate - down_rate) > 0.05:
            better = 'UP' if up_rate > down_rate else 'DOWN'
            insights.append(f"{better} trades perform better ({max(up_rate, down_rate):.1%} vs {min(up_rate, down_rate):.1%})")

    for i, insight in enumerate(insights, 1):
        print(f"    {i}. {insight}")

    if not insights:
        print("    Insufficient data for recommendations (need more trades)")

    print("\n" + "=" * 60)
    print("  Dashboard complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
