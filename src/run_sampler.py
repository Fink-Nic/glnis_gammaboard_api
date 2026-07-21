# type: ignore
from __future__ import annotations

import glnis_runtime_bootstrap  # noqa: F401
from gammaboard_process import run_sampler
from madnis_sampler import MadnisSampler


def main() -> None:
    run_sampler(MadnisSampler)


if __name__ == "__main__":
    main()
