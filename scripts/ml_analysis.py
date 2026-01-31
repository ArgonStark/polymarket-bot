#!/usr/bin/env python3
"""
ML Analysis Script

Run backtests, view feature importance, and analyze ML model performance.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategy.backtesting import get_backtester, run_monte_carlo
from src.strategy.ml_ensemble import get_enhanced_predictor
from src.strategy.ml_predictor import get_ml_predictor
from src.strategy.kelly import get_kelly_calculator


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def show_ml_stats():
    """Show current ML model statistics."""
    print_header("ML MODEL STATISTICS")

    # Basic ML predictor
    ml = get_ml_predictor()
    stats = ml.get_model_stats()

    print(f"\n📊 Basic ML Predictor:")
    print(f"   Model type: {stats.get('model_type', 'Unknown')}")
    print(f"   Training samples: {stats['training_samples']}")
    print(f"   Predictions made: {stats['predictions_made']}")
    print(f"   Correct predictions: {stats['correct_predictions']}")
    print(f"   Accuracy: {stats['accuracy']:.1%}")
    print(f"   Active: {stats['is_active']}")
    print(f"   Hybrid enabled: {stats.get('hybrid_enabled', False)}")

    # Enhanced predictor
    try:
        enhanced = get_enhanced_predictor()
        enhanced_stats = enhanced.get_stats()

        print(f"\n🚀 Enhanced ML Predictor:")
        print(f"   Training samples: {enhanced_stats['training_samples']}")
        print(f"   Ensemble accuracy: {enhanced_stats['ensemble']['ensemble_accuracy']:.1%}")
        print(f"   GBM samples: {enhanced_stats['gbm_samples']}")

        print(f"\n📈 Ensemble Models:")
        for model in enhanced_stats['ensemble']['models']:
            print(f"   - {model['name']}: accuracy={model['accuracy']:.1%}, weight={model['weight']:.2f}")

        print(f"\n🔍 Top Features by Importance:")
        for i, (name, importance) in enumerate(enhanced_stats['feature_importance'][:10], 1):
            print(f"   {i:2d}. {name}: {importance:.4f}")

    except Exception as e:
        print(f"\n   Enhanced predictor not available: {e}")


def show_kelly_stats():
    """Show Kelly criterion statistics."""
    print_header("KELLY CRITERION STATISTICS")

    kelly = get_kelly_calculator()
    stats = kelly.get_stats()

    print(f"\n💰 Kelly Calculator:")
    print(f"   Total bets: {stats['total_bets']}")
    print(f"   Wins: {stats['wins']}")
    print(f"   Win rate: {stats['win_rate']:.1%}")
    print(f"   Average edge: {stats['average_edge']:.2%}")
    print(f"   Kelly fraction: {stats['kelly_fraction']:.1%}")
    print(f"   Dynamic multiplier: {stats.get('dynamic_multiplier', 1.0):.2f}")
    print(f"   Expected growth: {stats['expected_growth']:.2%}")


def run_backtest():
    """Run a backtest on historical trades."""
    print_header("BACKTEST RESULTS")

    backtester = get_backtester()

    if not backtester.trades:
        print("\n❌ No historical trades found for backtesting.")
        print("   Trades will be loaded from trade_history.json")
        return

    print(f"\n📊 Running backtest on {len(backtester.trades)} trades...")

    # Run basic backtest
    result = backtester.run_backtest()

    print(f"\n📈 Performance Metrics:")
    print(f"   Initial bankroll: ${result.initial_bankroll:.2f}")
    print(f"   Final bankroll: ${result.final_bankroll:.2f}")
    print(f"   Total PnL: ${result.total_pnl:.2f}")
    print(f"   Total trades: {result.total_trades}")
    print(f"   Win rate: {result.win_rate:.1%}")
    print(f"   Profit factor: {result.profit_factor:.2f}")

    print(f"\n📉 Risk Metrics:")
    print(f"   Sharpe ratio: {result.sharpe_ratio:.2f}")
    print(f"   Sortino ratio: {result.sortino_ratio:.2f}")
    print(f"   Max drawdown: ${result.max_drawdown:.2f} ({result.max_drawdown_pct:.1%})")
    print(f"   Calmar ratio: {result.calmar_ratio:.2f}")

    print(f"\n📊 Trade Statistics:")
    print(f"   Average win: ${result.avg_win:.2f}")
    print(f"   Average loss: ${result.avg_loss:.2f}")
    print(f"   Largest win: ${result.largest_win:.2f}")
    print(f"   Largest loss: ${result.largest_loss:.2f}")
    print(f"   Max consecutive wins: {result.max_consecutive_wins}")
    print(f"   Max consecutive losses: {result.max_consecutive_losses}")


def run_monte_carlo_sim():
    """Run Monte Carlo simulation."""
    print_header("MONTE CARLO SIMULATION")

    backtester = get_backtester()

    if not backtester.trades:
        print("\n❌ No historical trades found for simulation.")
        return

    print(f"\n🎲 Running 1000 simulations...")

    results = run_monte_carlo(n_simulations=1000)

    if "error" in results:
        print(f"\n❌ Error: {results['error']}")
        return

    print(f"\n📊 Final Bankroll Distribution:")
    fb = results['final_bankroll']
    print(f"   Mean: ${fb['mean']:.2f}")
    print(f"   Median: ${fb['median']:.2f}")
    print(f"   Std Dev: ${fb['std']:.2f}")
    print(f"   5th percentile: ${fb['percentile_5']:.2f}")
    print(f"   25th percentile: ${fb['percentile_25']:.2f}")
    print(f"   75th percentile: ${fb['percentile_75']:.2f}")
    print(f"   95th percentile: ${fb['percentile_95']:.2f}")

    print(f"\n📉 Max Drawdown Distribution:")
    md = results['max_drawdown']
    print(f"   Mean: {md['mean']:.1%}")
    print(f"   Median: {md['median']:.1%}")
    print(f"   95th percentile (worst case): {md['percentile_95']:.1%}")

    print(f"\n📈 Win Rate Distribution:")
    wr = results['win_rate']
    print(f"   Mean: {wr['mean']:.1%}")
    print(f"   Std Dev: {wr['std']:.1%}")


def show_version_comparison():
    """Show model version A/B test comparison."""
    print_header("MODEL VERSION COMPARISON")

    try:
        enhanced = get_enhanced_predictor()
        comparison = enhanced.version_manager.get_comparison()

        print(f"\n🔬 Active Version: {comparison['active']['version_id']}")
        print(f"   Accuracy: {comparison['active']['accuracy']:.1%}")
        print(f"   Predictions: {comparison['active']['predictions']}")
        print(f"   Avg PnL: ${comparison['active']['avg_pnl']:.4f}")

        if 'challenger' in comparison:
            print(f"\n⚔️ Challenger Version: {comparison['challenger']['version_id']}")
            print(f"   Accuracy: {comparison['challenger']['accuracy']:.1%}")
            print(f"   Predictions: {comparison['challenger']['predictions']}")
            print(f"   Avg PnL: ${comparison['challenger']['avg_pnl']:.4f}")
            print(f"   Accuracy diff: {comparison['accuracy_diff']:+.1%}")
        else:
            print(f"\n   No challenger version set for A/B testing")

    except Exception as e:
        print(f"\n❌ Could not load version comparison: {e}")


def main():
    """Main entry point."""
    print("\n" + "🤖 POLYMARKET BOT - ML ANALYSIS" + "\n")

    # Show all stats
    show_ml_stats()
    show_kelly_stats()
    run_backtest()
    run_monte_carlo_sim()
    show_version_comparison()

    print("\n" + "=" * 60)
    print("  Analysis complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
