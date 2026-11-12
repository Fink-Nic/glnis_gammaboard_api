# type: ignore
from __future__ import annotations

from gammaboard_process import run_sampler
from gammaboard_sampler import GammaBoardSampler


def main() -> None:
    run_sampler(GammaBoardSampler)


if __name__ == "__main__":
    main()
