"""Predeclared daily-bar search spaces; no label-dependent bounds."""
from dataclasses import asdict, dataclass
from decimal import Decimal
import random

from wyckoff_core.models import CoreConfig

SPACE_VERSION = "daily-v1"
JOINT_SEARCH_VERSION = "daily-joint-v1"
BATCHES = {
    "A": ("swing_confirmation", "min_range_bars", "min_range_tests", "boundary_buffer_atr", "test_tolerance_atr"),
    "B": ("baseline_window", "trend_window", "volume_high", "volume_low", "spread_high", "trend_min_atr"),
    "C": ("phase_e_closes",),
    "D1": ("core_atr_period", "local_trend_bars"),
    "D2": ("range_max_pct", "range_min_height_atr"),
    "D3": ("preliminary_volume_min", "preliminary_spread_min", "climax_recovery_close", "climax_sequence_max_bars", "ar_min_move_atr", "st_volume_ratio_max"),
    "D4": ("interior_test_spread_max", "probe_volume_ratio_max", "probe_test_spread_max"),
    "D5": ("lps_volume_ratio_max", "lps_spread_ratio_max", "lpsy_poor_result_max_atr", "initiative_spread_min", "initiative_close_min", "initiative_move_min_atr"),
}
JOINT_BATCHES = {
    "J1": ("swing_confirmation", "min_range_tests", "test_tolerance_atr", "trend_window",
           "range_min_height_atr", "range_max_pct"),
    "J2": ("baseline_window", "volume_high", "volume_low", "initiative_spread_min",
           "initiative_close_min", "initiative_move_min_atr"),
    "J3": ("volume_low", "interior_test_spread_max", "probe_volume_ratio_max",
           "probe_test_spread_max", "lps_volume_ratio_max", "lps_spread_ratio_max"),
}


@dataclass(frozen=True)
class Dimension:
    choices: tuple = ()
    low: float | None = None
    high: float | None = None
    step: float | None = None

    def values(self):
        if self.choices:
            return self.choices
        a, b, step = (Decimal(str(x)) for x in (self.low, self.high, self.step))
        if (b - a) % step:
            raise ValueError("search endpoint is not aligned with step")
        return tuple(float(a + i * step) for i in range(int((b - a) / step) + 1))

    def suggest(self, trial, name):
        if self.choices:
            return trial.suggest_categorical(name, list(self.choices))
        return trial.suggest_float(name, self.low, self.high, step=self.step)


def search_space(market):
    if market not in ("us_equities", "binance_spot"):
        raise ValueError("unsupported market")
    coin = market == "binance_spot"
    enum = lambda stock, crypto=None: Dimension(tuple(crypto if coin and crypto is not None else stock))
    linear = lambda a, b, step, crypto_low=None, crypto_high=None: Dimension(
        low=crypto_low if coin and crypto_low is not None else a,
        high=crypto_high if coin and crypto_high is not None else b, step=step)
    quiet = enum((0.8, 1.0, 1.2, 1.4), (0.8, 1.0, 1.2, 1.5))
    return {
        "baseline_window": enum((15, 20, 30, 40), (20, 28, 42, 56)),
        "trend_window": enum((10, 15, 20, 30), (14, 15, 21, 28, 42)),
        "swing_confirmation": enum((1, 2, 3, 4), (2, 3, 4, 5)),
        "min_range_bars": enum((8, 10, 15, 20, 30), (10, 14, 21, 28, 42)),
        "min_range_tests": enum((2, 3, 4)),
        "volume_high": linear(1.2, 2.2, 0.1, crypto_high=2.5),
        "volume_low": linear(0.6, 1.0, 0.05, crypto_low=0.5),
        "spread_high": linear(1.2, 2.2, 0.1, crypto_high=2.5),
        "boundary_buffer_atr": enum((0.05, 0.10, 0.20, 0.30), (0.05, 0.10, 0.20, 0.35, 0.50)),
        "test_tolerance_atr": enum((0.5, 0.75, 1.0, 1.25, 1.5), (0.5, 0.75, 1.0, 1.5, 2.0)),
        "trend_min_atr": enum((0.5, 0.75, 1.0, 1.5, 2.0), (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)),
        "phase_e_closes": enum((2, 3, 5, 8), (3, 5, 7, 10)),
        "core_atr_period": enum((10, 14, 20, 28), (14, 21, 28, 42)),
        "local_trend_bars": enum((3, 5, 10), (5, 7, 14)),
        "range_max_pct": enum((0.20, 0.35, 0.50), (0.25, 0.35, 0.50, 0.75)),
        "range_min_height_atr": enum((0.75, 1.0, 1.5, 2.0), (1.0, 1.5, 2.0, 3.0)),
        "preliminary_volume_min": linear(1.05, 1.40, 0.05, crypto_high=1.60),
        "preliminary_spread_min": linear(1.05, 1.40, 0.05, crypto_high=1.60),
        "climax_recovery_close": enum((0.50, 0.55, 0.60), (0.50, 0.55, 0.60, 0.65)),
        "climax_sequence_max_bars": enum((40, 60, 90, 120), (42, 60, 90, 120)),
        "ar_min_move_atr": enum((0.75, 1.0, 1.5, 2.0), (1.0, 1.5, 2.0, 3.0)),
        "st_volume_ratio_max": quiet,
        "interior_test_spread_max": quiet,
        "probe_volume_ratio_max": quiet,
        "probe_test_spread_max": quiet,
        "lps_volume_ratio_max": quiet,
        "lps_spread_ratio_max": quiet,
        "lpsy_poor_result_max_atr": enum((0.25, 0.5, 0.75)),
        "initiative_spread_min": enum((1.0, 1.2, 1.5, 1.8), (1.0, 1.2, 1.5, 2.0)),
        "initiative_close_min": enum((0.60, 0.65, 0.70, 0.75, 0.80)),
        "initiative_move_min_atr": enum((0.25, 0.5, 0.75, 1.0), (0.25, 0.5, 0.75, 1.0, 1.5)),
    }


def validate_candidate(params, market):
    cfg = CoreConfig(**params)
    for name, value in asdict(cfg).items():
        if not any(abs(value - x) <= 1e-10 for x in search_space(market)[name].values()):
            raise ValueError(f"{name} is outside {SPACE_VERSION} search space")
    if not cfg.volume_low <= 1 < cfg.volume_high:
        raise ValueError("volume_low <= 1 < volume_high required")
    if cfg.min_range_bars < 2 * cfg.swing_confirmation + 1:
        raise ValueError("range length cannot be below confirmation span")
    if cfg.local_trend_bars > cfg.trend_window:
        raise ValueError("local trend cannot exceed trend window")
    if max(cfg.min_range_bars, cfg.trend_window) > cfg.climax_sequence_max_bars:
        raise ValueError("climax sequence shorter than required windows")
    if cfg.preliminary_volume_min > cfg.volume_high or cfg.preliminary_spread_min > cfg.spread_high:
        raise ValueError("preliminary thresholds exceed climax thresholds")
    if max(cfg.st_volume_ratio_max, cfg.probe_volume_ratio_max, cfg.lps_volume_ratio_max) > cfg.volume_high:
        raise ValueError("quiet-volume threshold exceeds high-volume threshold")
    return cfg


def sample_configs(market, count=32, seed=20260930, batches=None):
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 4096:
        raise ValueError("samples must be an integer from 0 to 4096")
    dimensions = search_space(market)
    declared = {**BATCHES, **JOINT_BATCHES}
    names = tuple(dimensions) if batches is None else tuple(dict.fromkeys(n for b in batches for n in declared[b]))
    rng = random.Random(seed)
    results = []
    for _ in range(count):
        for attempt in range(1000):
            params = asdict(CoreConfig())
            params.update({name: rng.choice(dimensions[name].values()) for name in names})
            try:
                results.append(validate_candidate(params, market))
                break
            except ValueError:
                if attempt == 999:
                    raise ValueError("cannot sample a feasible configuration")
    return results


def common_warmup(market):
    space = search_space(market)
    return max(max(space[k].values()) for k in ("baseline_window", "trend_window", "core_atr_period")) + 1
