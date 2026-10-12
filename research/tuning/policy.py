"""Promotion gates registered before supervised search, never fitted to test scores."""

VALIDATION_POLICY = {
    "version": "labeled-holdout-v1",
    "bootstrap_seed": 20260930,
    "bootstrap_repeats": 2000,
    "confidence_level": 0.95,
    "minimum_effective_clusters_per_panel": 12,
    "neighbor_acceptable_fraction": 0.9,
    "maximum_development_regression": 0.05,
    "minimum_holdout_f1_improvement": 0.0,
    "minimum_holdout_agreement_improvement": 0.0,
    "scope": "One security and declared label mode; labeled samples only.",
    "limitations": "Engineering eligibility thresholds, not a power calculation. "
        "Nominal per-security intervals; no simultaneous claim across the universe.",
}
