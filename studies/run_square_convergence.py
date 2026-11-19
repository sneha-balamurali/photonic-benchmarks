import csv
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt

from examples.model_square import model
from src.benchmark import run_backends
from src.config import Config

import json

from src.fmmax.config import FMMaxConfig
from src.metarcwa.config import MetaRCWAConfig
from src.s4.config import S4Config


def run_square_convergence(config: Config, output_directory: Path, *, study_name: str) -> None:
    """Run the square convergence sweep and save its CSV and plot."""
    backends = {
        "fmmax": ("FMMax", "x", "#0072B2"),
        "s4": ("S4", "o", "#D55E00"),
        "metarcwa": ("MetaRCWA", "^", "#009e73"),
    }
    columns = [
        "study", "backend", "m", "n", "nx", "ny",
        "requested_orders", "actual_orders",
        "Rs", "Rp", "Ts", "Tp", "formulation_settings",
    ]
    rows = []

    for order in range(1, 11):
        current_config = replace(config, m=order, n=order)

        s4 = S4Config.from_config(current_config)
        fmmax = FMMaxConfig.from_config(current_config)
        metarcwa = MetaRCWAConfig.from_config(current_config)

        settings = {
            "s4": {
                "polarization_decomposition": s4.polarization_decomposition,
                "polarization_basis": s4.polarization_basis,
                "discretized_epsilon": s4.discretized_epsilon,
                "discretization_resolution": s4.discretization_resolution,
            },
            "fmmax": {"formulation": fmmax.formulation.value},
            "metarcwa": {
                "factorization": (
                    metarcwa.factorization.to_dict()
                    if metarcwa.factorization is not None
                    else None
                ),
            },
        }

        results = run_backends(model, current_config)

        for backend in backends:
            result = results[backend]
            rows.append({
                "study": study_name,
                "backend": backend,
                "m": current_config.m,
                "n": current_config.n,
                "nx": current_config.nx,
                "ny": current_config.ny,
                "requested_orders": int(result["requested_orders"]),
                "actual_orders": int(result["actual_orders"]),
                "Rs": float(result["Rs"][0, 0, 0]),
                "Rp": float(result["Rp"][0, 0, 0]),
                "Ts": float(result["Ts"][0, 0, 0]),
                "Tp": float(result["Tp"][0, 0, 0]),
                "formulation_settings": json.dumps(settings[backend], sort_keys=True),
            })

    output_directory.mkdir(parents=True, exist_ok=True)

    csv_path = output_directory / "square_convergence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    for backend, (label, marker, color) in backends.items():
        backend_rows = [row for row in rows if row["backend"] == backend]
        ax.plot(
            [row["actual_orders"] for row in backend_rows],
            [row["Rs"] for row in backend_rows],
            marker=marker,
            color=color,
            label=label,
        )

    ax.set_xlabel(r"Actual number of retained Fourier harmonics, $N_h$")
    ax.set_ylabel(r"$R_s$")
    ax.set_title(f"{study_name}\nSquare-particle s-polarised reflectance convergence with fourier basis")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    png_path = output_directory / "square_convergence.png"
    fig.savefig(png_path, dpi=200)
    print(f"Saved {csv_path}")
    print(f"Saved {png_path}")
    plt.show()