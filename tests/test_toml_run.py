"""End-to-end smoke coverage for the user-facing TOML test script."""

import json
from pathlib import Path
import subprocess
import sys


def test_script_runs_default_toml_from_another_directory_and_exports_history(tmp_path):
    script = Path(__file__).with_name("run_from_toml.py").resolve()
    output = tmp_path / "history.json"
    completed = subprocess.run(
        [sys.executable, str(script), "--history", str(output)],
        cwd=tmp_path, text=True, capture_output=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Completed 10 training steps and 2048 inference samples" in completed.stdout
    assert completed.stdout.count("Training step") == 1
    history = json.loads(output.read_text())
    assert [record["step"] for record in history["training"]] == list(range(1, 11))
    assert all(record["n_samples"] == 128 for record in history["training"])
    assert len(history["inference"]) == 64
    assert history["inference"][-1]["n_samples"] == 2048
    assert sum(record["update_samples"] for record in history["inference"]) == 2048
    assert history["inference"][-1]["observables"]["n_failures"] == 0
