import csv
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt

from examples.model_square import model
from src.benchmark import run_backends
from src.config import Config


def run_square_convergence(config: Config, output_directory: Path) -> None:
    """Run the square convergence sweep and save its CSV and plot."""
    backends = {
        "fmmax": ("FMMax", "x"),
        "s4": ("S4", "o"),
        "metarcwa": ("MetaRCWA", "^"),
    }
    columns = [
        "backend", "m", "n", "requested_orders", "actual_orders",
        "Rs", "Rp", "Ts", "Tp",
    ]
    rows = []

    for order in range(1, 11):
        current_config = replace(config, m=order, n=order)
        results = run_backends(model, current_config)

        for backend in backends:
            result = results[backend]
            rows.append({
                "backend": backend,
                "m": current_config.m,
                "n": current_config.n,
                "requested_orders": int(result["requested_orders"]),
                "actual_orders": int(result["actual_orders"]),
                "Rs": float(result["Rs"][0, 0, 0]),
                "Rp": float(result["Rp"][0, 0, 0]),
                "Ts": float(result["Ts"][0, 0, 0]),
                "Tp": float(result["Tp"][0, 0, 0]),
            })

    output_directory.mkdir(parents=True, exist_ok=True)

    csv_path = output_directory / "square_convergence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    for backend, (label, marker) in backends.items():
        backend_rows = [row for row in rows if row["backend"] == backend]
        ax.plot(
            [row["actual_orders"] for row in backend_rows],
            [row["Rs"] for row in backend_rows],
            marker=marker,
            label=label,
        )

    ax.set_xlabel(r"Actual number of retained Fourier harmonics, $N_h$")
    ax.set_ylabel(r"$R_s$")
    ax.set_title(r"Total $s$-polarized reflectance vs retained basis size")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    png_path = output_directory / "square_convergence.png"
    fig.savefig(png_path, dpi=200)
    print(f"Saved {csv_path}")
    print(f"Saved {png_path}")
    plt.show()