# type: ignore
from madnis_sampler import MadnisSampler
from gammaboard_process import run_sampler


def main() -> None:
    run_sampler(MadnisSampler)


if __name__ == "__main__":
    main()
