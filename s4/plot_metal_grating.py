"""Plot convergence and CPU time for the S4 metal grating benchmark."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIRECTORY = Path("results")
CSV_PATH = RESULTS_DIRECTORY / "s4_metal_grating_convergence.csv"

def read_results():
    """Read the S4 benchmark results from the CSV file."""
    results = pd.read_csv(CSV_PATH)
    return results

def plot_convergence(results):
    """Plot s- and p-polarized reflectance against S4 Fourier basis size."""
    plt.figure()

    for formulation, formulation_results in results.groupby("form"):
        plt.plot(
            formulation_results["s4_num_g"],
            formulation_results["R_s"],
            marker="o",
            label=f"s - {formulation}",
        )

        plt.plot(
            formulation_results["s4_num_g"],
            formulation_results["R_p"],
            marker="x",
            linestyle="--",
            label=f"p - {formulation}"
        )

    plt.xlabel("S4 number of x-directed Fourier harmonics (NumG)")
    plt.ylabel("Reflectance")
    plt.title("S4 Metal-Grating Reflectance Convergence with Fourier Harmonics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = RESULTS_DIRECTORY / "s4_reflectance_convergence.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved convergence plot to {output_path}")

def plot_cpu_time(results):
    """Plot s and p CPU time against S4 Fourier basis size."""

    plt.figure()

    for formulation, formulation_results in results.groupby("form"):
        plt.plot(
            formulation_results["s4_num_g"],
            formulation_results["s_cpu_seconds"],
            marker="o",
            label = "s"
        )

        plt.plot(
            formulation_results["s4_num_g"],
            formulation_results["p_cpu_seconds"],
            marker="x",
            linestyle="--",
            label = "p"
        )

        plt.xlabel("S4 number of x-directed Fourier harmonics (NumG)")
        plt.ylabel("s and p CPU time (seconds)")
        plt.title("S4 Metal-Grating CPU Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        output_path = RESULTS_DIRECTORY / "s4_cpu_time.png"
        plt.savefig(output_path, dpi=200)
        plt.close()

    print(f"Saved CPU time plot to {output_path}")


def main() -> None:
    """Load benchmark data and produce all plots."""

    RESULTS_DIRECTORY.mkdir(exist_ok=True)

    results = read_results()
    plot_convergence(results)
    plot_cpu_time(results)


if __name__ == "__main__":
    main()