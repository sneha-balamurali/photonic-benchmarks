from pathlib import Path

import yaml

from src.config import Config
from studies.run_square_convergence import run_square_convergence


def main() -> None:
    root = Path(__file__).resolve().parents[2]

    with (root / "src" / "config.yaml").open(encoding="utf-8") as file:
        config = Config.from_dict(yaml.safe_load(file))

    output_directory = root / "outputs" / "factorisation" / "jones" / "sampled_2"
    if output_directory.exists():
        raise FileExistsError(f"Choose a fresh output folder: {output_directory}")

    run_square_convergence(config, output_directory, study_name="Sampled Jones")


if __name__ == "__main__":
    main()