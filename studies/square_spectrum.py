"""Fourier-order convergence at the square model's fixed wavelength and angles.

Run from the repository root with:
    python -m studies.square_spectrum
    python -m studies.square_spectrum --max-order 4

The YAML m=n setting is the default sweep endpoint. Spatial resolution stays
fixed; this study does not establish convergence of the FMMax raster grid.
"""

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from src.benchmark import run_backends
from src.config import Config
from src.model.model_square import model


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "src/config.yaml")
    parser.add_argument("--max-order", type=int, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    with args.config.open() as stream:
        config = Config.from_dict(yaml.safe_load(stream))

    if args.max_order is None and config.m != config.n:
        parser.error("This square sweep requires YAML m=n or --max-order.")
    max_order = config.m if args.max_order is None else args.max_order
    if max_order < 1:
        parser.error("The maximum order must be at least 1 for a convergence sweep.")

    for name in ("wavelength", "theta", "phi"):
        if getattr(model.source, name).numel() != 1:
            parser.error(f"Convergence requires exactly one source {name}.")

    rows = []
    for order in range(max_order + 1):
        print(f"Running m=n={order}...", flush=True)
        results = run_backends(model, replace(config, m=order, n=order))
        for backend, result in results.items():
            row = {
                "backend": backend,
                "m": order,
                "n": order,
                "requested_orders": result["requested_orders"],
                "actual_orders": result["actual_orders"],
                **{key: float(result[key][0, 0, 0]) for key in ("Rs", "Rp", "Ts", "Tp")},
            }
            rows.append(row)
            print(f"  {backend}: orders={row['actual_orders']}, Rs={row['Rs']:.8f}", flush=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for backend in ("fmmax", "s4"):
        curve = [row for row in rows if row["backend"] == backend]
        ax.plot(
            [row["actual_orders"] for row in curve],
            [row["Rs"] for row in curve],
            marker="o", linewidth=1.8, label=backend.upper(),
        )
    wavelength = float(model.source.wavelength.item())
    ax.set_xlabel("Actual Fourier orders")
    ax.set_ylabel("Total s-polarised reflectance")
    ax.set_title(f"Square particle convergence at {wavelength:g} nm")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "square_convergence.png"
    data_path = output_dir / "square_convergence.csv"
    fig.savefig(image_path, dpi=200)
    with data_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved plot to {image_path}\nSaved data to {data_path}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
