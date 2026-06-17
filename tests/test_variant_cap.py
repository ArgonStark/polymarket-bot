"""Tests for per-variant position cap (TradingConfig.variant_position_cap)."""
from src.config import TradingConfig


def _cfg(max_concurrent: int, variants: list, explicit: int = 0) -> TradingConfig:
    cfg = TradingConfig()
    cfg.max_concurrent_positions = max_concurrent
    cfg.trading_variants = variants
    cfg.max_positions_per_variant = explicit
    return cfg


class TestVariantPositionCap:
    def test_auto_two_variants_two_slots(self):
        """2 slots / 2 variants → 1 reserved each."""
        assert _cfg(2, ["fifteen", "five"]).variant_position_cap() == 1

    def test_auto_two_variants_three_slots(self):
        """3 slots / 2 variants → ceil → 2 per variant (2+1 split possible)."""
        assert _cfg(3, ["fifteen", "five"]).variant_position_cap() == 2

    def test_auto_single_variant_no_restriction(self):
        """One variant → cap equals total, behavior unchanged."""
        assert _cfg(2, ["fifteen"]).variant_position_cap() == 2
        assert _cfg(4, ["fifteen"]).variant_position_cap() == 4

    def test_explicit_override_wins(self):
        assert _cfg(4, ["fifteen", "five"], explicit=3).variant_position_cap() == 3

    def test_never_below_one(self):
        assert _cfg(1, ["fifteen", "five"]).variant_position_cap() == 1
