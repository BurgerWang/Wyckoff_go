"""Shared contracts for descriptive Wyckoff evidence, not trading signals."""
from dataclasses import dataclass, fields
import math

CORE_VERSION = "3.0.5"


@dataclass(frozen=True)
class CoreConfig:
    """Engineering thresholds. They are not universal Wyckoff constants.

    v3 keeps the same 31 controls: ``phase_e_closes`` also measures prior
    separation for a new stop; ``trend_window`` bounds stop-to-AR waiting;
    ``local_trend_bars`` bounds recoveries and displacement waves; and
    ``climax_sequence_max_bars`` bounds unfinished sequences and the minimum
    age of a separated, unsupported range before retirement. All are explicit
    heuristics, not independently calibrated statistical facts.
    """
    baseline_window: int = 20
    trend_window: int = 15
    swing_confirmation: int = 2
    min_range_bars: int = 10
    min_range_tests: int = 2
    volume_high: float = 1.5
    volume_low: float = 0.8
    spread_high: float = 1.5
    boundary_buffer_atr: float = 0.10
    test_tolerance_atr: float = 1.0
    trend_min_atr: float = 1.0
    phase_e_closes: int = 3
    core_atr_period: int = 14
    local_trend_bars: int = 5
    range_max_pct: float = 0.35
    range_min_height_atr: float = 1.0
    preliminary_volume_min: float = 1.2
    preliminary_spread_min: float = 1.2
    climax_recovery_close: float = 0.5
    climax_sequence_max_bars: int = 60
    ar_min_move_atr: float = 1.0
    st_volume_ratio_max: float = 1.2
    interior_test_spread_max: float = 1.2
    probe_volume_ratio_max: float = 1.2
    probe_test_spread_max: float = 1.2
    lps_volume_ratio_max: float = 1.2
    lps_spread_ratio_max: float = 1.2
    lpsy_poor_result_max_atr: float = 0.5
    initiative_spread_min: float = 1.2
    initiative_close_min: float = 0.7
    initiative_move_min_atr: float = 0.5

    def __post_init__(self):
        integers = ("baseline_window", "trend_window", "swing_confirmation", "min_range_bars",
                    "min_range_tests", "phase_e_closes", "core_atr_period", "local_trend_bars",
                    "climax_sequence_max_bars")
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{item.name} must be positive and finite")
            if item.name in integers and not isinstance(value, int):
                raise ValueError(f"{item.name} must be a positive integer")
        if (self.baseline_window < 2 or self.trend_window < 3 or self.volume_low >= self.volume_high
                or self.min_range_tests < 2 or self.min_range_bars < 3):
            raise ValueError("invalid feature windows or volume thresholds")
        if self.range_max_pct > 1:
            raise ValueError("range_max_pct must be in (0, 1]")
        if not 0.5 <= self.climax_recovery_close <= 1:
            raise ValueError("climax_recovery_close must be in [0.5, 1]")
        if not 0.5 < self.initiative_close_min <= 1:
            raise ValueError("initiative_close_min must be in (0.5, 1]")


@dataclass
class TR:
    """Frozen price boundaries; lifecycle and side are separate evidence."""
    start_col: int
    end_col: int
    top: float
    bottom: float
    width: int
    mid: float
    bias: str = "unknown"
    avg_volume: float = 0.0
    trend_bars: int = 0
    start_date: object = None
    end_date: object = None
    confirmed_date: object = None
    established_date: object = None
    cause_end_date: object = None
    range_id: str = ""
    origin: str = "pnf_congestion"
    lifecycle: str = "active"
    parent_range_id: str | None = None

    @property
    def range_pct(self):
        return (self.top - self.bottom) / self.bottom if self.bottom else 0.0
