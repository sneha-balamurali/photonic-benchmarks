"""Compare FMMax and S4 metal-grating convergence results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIRECTORY = Path("results")
FMMAX_CSV = RESULTS_DIRECTORY / "metal_grating_benchmark.csv"
S4_CSV = RESULTS_DIRECTORY / "s4_metal_grating_convergence.csv"

def main() -> None:
    fmmax = pd.read_csv(FMMAX_csv)
    s4 = pd.read_csv(S4_CSV)

    