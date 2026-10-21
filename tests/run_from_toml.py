"""Runnable smoke script: python tests/run_from_toml.py [run.toml] [--history out.json]."""

import os
from pathlib import Path

# Small CPU smoke runs are faster without a large BLAS/OpenMP thread pool.
# Preserve explicit settings supplied by the caller.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from glnis.scripts.run import main


if __name__ == "__main__":
    main(default_config=Path(__file__).resolve().parents[1] / "examples/local/ghostbumps.toml")
