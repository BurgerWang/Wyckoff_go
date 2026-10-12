import json
import subprocess
import sys

import numpy as np
import pandas as pd

from analysis import analyze_frame, main, to_json


def bars():
    close = 100 + 5 * np.sin(np.arange(90) / 3)
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1,
                         "Close": close, "Volume": 100.0},
                        index=pd.date_range("2025-01-01", periods=90))


def test_snapshot_all_scales_is_serializable_and_does_not_mutate_input():
    df = bars()
    original = df.copy()
    result = analyze_frame(df, "TEST", ["atr", "pct", "trad"])
    pd.testing.assert_frame_equal(df, original)
    assert df.attrs == {}
    data = json.loads(to_json(result))
    assert data["schema_version"] == 2 and set(data["analyses"]) == {"atr", "pct", "trad"}
    for analysis in result["analyses"].values():
        assert analysis["wyckoff"]["as_of"] == df.index[-1]
        assert all(e["date"] <= e["available_date"] <= df.index[-1] for e in analysis["wyckoff"]["events"])


def test_csv_cli_writes_real_report_and_returns_nonzero_on_invalid_bars(tmp_path, capsys):
    source = tmp_path / "input.csv"
    target = tmp_path / "report.json"
    bars().to_csv(source, index_label="Date")
    assert main(["TEST", "--csv", str(source), "--format", "json", "--output", str(target)]) == 0
    assert json.loads(target.read_text())["symbol"] == "TEST"
    assert "schema_version" in capsys.readouterr().out
    bars().iloc[::-1].to_csv(source, index_label="Date")
    assert main(["--csv", str(source)]) == 2
    assert "strictly increasing" in capsys.readouterr().err


def test_core_imports_do_not_load_optional_render_or_download_libraries():
    result = subprocess.run([sys.executable, "-c", "import pnf, wyckoff, market_data, analysis; import sys; assert 'matplotlib' not in sys.modules; assert 'yfinance' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
