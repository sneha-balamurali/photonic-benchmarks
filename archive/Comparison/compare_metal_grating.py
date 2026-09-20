"""Compare FMMax and S4 metal-grating convergence results."""

from pathlib import Path

# For creating the plots
import matplotlib.pyplot as plt
# For reading the CSV files and representing them as spreadsheet-like
# DataFrame objects 
import pandas as pd


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = SCRIPT_DIRECTORY.parent
RESULTS_DIRECTORY = Path("results")

RESULTS_DIRECTORY.mkdir(exist_ok=True)

FMMAX_CSV =  PROJECT_DIRECTORY / "fmmax" / "results" / "metal_grating_comparison_1.csv"
S4_CSV = PROJECT_DIRECTORY / "s4" / "results" / "s4_metal_grating_comparison_1.csv"

def main() -> None:
    fmmax = pd.read_csv(FMMAX_CSV)
    s4 = pd.read_csv(S4_CSV)

    # FMM FTT and parallelogramic truncation vs S4 FFT
    fmmax_fft = fmmax[(fmmax["formulation"] == "fft")
    & (fmmax["truncation"] == "parallelogramic")].sort_values("num_terms")

    s4_fft = s4[s4["form"] == "fft"].sort_values("fmmax_equivalent_terms")

    plot_te_convergence(fmmax_fft, s4_fft)
    plot_tm_convergence(fmmax_fft, s4_fft)

def plot_te_convergence(
    fmmax: pd.DataFrame,
    s4: pd.DataFrame,
    ) -> None:
    plt.figure()

    plt.plot(
            fmmax["num_terms"],
            fmmax["R_te"],
            marker="o",
            label="FMMAX FFT",
            )
    
    plt.plot(
        s4["fmmax_equivalent_terms"],
        s4["R_s"],
        marker="x",
        linestyle="--",
        label="S4 FFT",
    )

    plt.xlabel("Equivalent Fourier basis target")
    plt.ylabel("TE specular reflectance")
    plt.title("Metal-Grating TE Convergence: FMMAX vs S4")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = RESULTS_DIRECTORY / "compare_te_convergence_1.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_tm_convergence(
    fmmax: pd.DataFrame,
    s4: pd.DataFrame,
    ) -> None:
    plt.figure()

    plt.plot(
        fmmax["num_terms"],
        fmmax["R_tm"],
        marker="o",
        label="FMMax FFT",
    )

    plt.plot(
        s4["fmmax_equivalent_terms"],
        s4["R_p"],
        marker="x",
        linestyle="--",
        label="S4 FFT",
    )

    plt.xlabel("Equivalent Fourier basis target")
    plt.ylabel("TM Specular reflectance")
    plt.title("Metal-Grating TM Convergene: FMMax vs S4")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = RESULTS_DIRECTORY / "compare_tm_convergence_1.png"
    plt.savefig(output_path, dpi = 200)
    plt.close()

    print(f"Saved{output_path}")

if __name__ == "__main__":
    main()