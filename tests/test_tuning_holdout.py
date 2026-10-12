"""Exposure guards persist across directories, snapshots and label modes."""
from copy import deepcopy

import pytest

from research.tuning import holdout


@pytest.fixture
def reference(tmp_path, monkeypatch):
    monkeypatch.setattr(holdout, "REGISTRY", tmp_path / "shared-holdouts")
    return {"market": "binance_spot", "symbol": "BTCUSDT", "label_mode": "asof",
        "labels": [{"as_of": "2025-02-03", "panel": "window", "block_id": "test-block",
                    "information_start": "2025-01-01", "information_end": "2025-02-03"}]}


def test_claim_blocks_other_owner_mode_and_renamed_groups(reference):
    with holdout.claim_holdout(reference, {"run_signature": "engineering-a"}):
        pass
    other = deepcopy(reference)
    other["label_mode"] = "retrospective"
    other["labels"][0]["block_id"] = "renamed"
    other["source_snapshot"] = "appended-data"
    with pytest.raises(ValueError, match="another frozen candidate"):
        with holdout.claim_holdout(other, {"run_signature": "engineering-b"}):
            pass
    with pytest.raises(ValueError, match="already exposed"):
        holdout.check_unused(other)
    with holdout.claim_holdout(reference, {"run_signature": "engineering-a"}):
        pass
    holdout.check_owner(reference, {"run_signature": "engineering-a"})


def test_failed_evaluation_retains_exposure_for_identical_recovery(reference):
    with pytest.raises(RuntimeError):
        with holdout.claim_holdout(reference, {"run_signature": "engineering-a"}):
            raise RuntimeError("simulated scoring failure")
    with pytest.raises(ValueError):
        holdout.check_unused(reference)
    with holdout.claim_holdout(reference, {"run_signature": "engineering-a"}):
        pass


def test_nonoverlapping_information_can_be_new_holdout(reference):
    with holdout.claim_holdout(reference, {"run_signature": "engineering-a"}):
        pass
    later = deepcopy(reference)
    later["labels"][0].update(as_of="2025-06-01", information_start="2025-05-01", information_end="2025-06-01")
    holdout.check_unused(later)
    with holdout.claim_holdout(later, {"run_signature": "engineering-b"}):
        pass
    with pytest.raises(ValueError, match="ownership"):
        holdout.check_owner(later, {"run_signature": "engineering-a"})


def test_shared_target_lock_prevents_concurrent_different_runs(reference):
    with holdout.claim_holdout(reference, {"run_signature": "engineering-a"}):
        with pytest.raises(ValueError, match="already writing"):
            with holdout.claim_holdout(reference, {"run_signature": "engineering-b"}):
                pass
