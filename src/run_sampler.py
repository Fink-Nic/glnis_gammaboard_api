# type: ignore
from __future__ import annotations

import glnis_runtime_bootstrap

glnis_runtime_bootstrap.bootstrap()

from gammaboard_process import run_sampler  # noqa: E402
from madnis_sampler import MadnisSampler  # noqa: E402


def main() -> None:
    run_sampler(MadnisSampler)


if __name__ == "__main__":
    main()
