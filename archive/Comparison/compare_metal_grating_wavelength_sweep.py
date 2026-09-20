"""Compare FMMax and S4 TE wavelength sweeps."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FMMAX_RESULTS = (
    Path("../fmmax/results/metal_grating_spectrum.csv")
)

S4_RESULTS = (
    Path("../s4/results/s4_metal_grating_spectrum.csv")
)


def main():

    fmmax = pd.read_csv(FMMAX_RESULTS)
    s4 = pd.read_csv(S4_RESULTS)

    plt.figure(figsize=(8,5))

    plt.plot(
        fmmax["wavelength_nm"],
        fmmax["R_te"],
        marker="o",
        label="FMMax",
    )

    plt.plot(
        s4["wavelength_nm"],
        s4["R_s"],
        marker="x",
        linestyle="--",
        label="S4",
    )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("TE Reflectance")
    plt.title("Metal Grating Wavelength Sweep")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output = Path("results/compare_te_wavelength_sweep.png")
    output.parent.mkdir(exist_ok=True)

    plt.savefig(output, dpi=200)
    plt.close()

    print(f"Saved {output}")


if __name__ == "__main__":
    main()