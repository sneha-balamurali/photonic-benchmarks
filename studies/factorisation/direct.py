# Formulation Family: Direct
# S4: Will comapre both Analytic rectangle 
# Discretised epsilon, polarisation decomposition off
# MetaRCWA: factorization = None
# FMMax: FFT

from pathlib import Path

import yaml

from src.config import Config
from studies.run_square_convergence import run_square_convergence


def main() -> None:
    root = Path(__file__).resolve().parents[2]

    with (root / "src" / "config.yaml").open(encoding="utf-8") as file:
        config = Config.from_dict(yaml.safe_load(file))

    output_directory = root / "outputs" / "factorisation" / "direct" / "sampled"
    run_square_convergence(config, output_directory)


if __name__ == "__main__":
    main()