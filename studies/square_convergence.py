from pathlib import Path
import matplotlib.pyplot as plt
import yaml
from src.benchmark import run_backends
from src.config import Config
from src.model.model_square import model
import csv

# Load the YAML configuration 
with open("src/config.yaml") as file:
    config = Config.from_dict(yaml.safe_load(file)) 

# Store the values we want to plot
fmmax_orders = []
fmmax_Rs = []

s4_orders = []
s4_Rs = []

metarcwa_orders = []
metarcwa_Rs = []

csv_rows = []

# Repeat the calculation for different Fourier orders
for order in [1,2,3,4,5,6,7,8,9,10]:
    current_config = Config.from_dict({
        **config.to_dict(),
        "m": order,
        "n": order
    })

    results = run_backends(model, current_config)

    # Add three rows: one per solver for each fourier setting
    for backend in ("fmmax", "s4", "metarcwa"):
        backend_results = results[backend]

        row = {
            "backend": backend,
            "m": current_config.m,
            "n": current_config.n,
            "requested_orders": int(backend_results["requested_orders"]),
            "actual_orders": int(backend_results["actual_orders"]),
            "Rs": float(backend_results["Rs"][0,0,0]),
            "Rp": float(backend_results["Rp"][0,0,0]),
            "Ts": float(backend_results["Ts"][0,0,0]),
            "Tp": float(backend_results["Tp"][0,0,0])
        }

        csv_rows.append(row)

    fmmax_orders.append(results["fmmax"]["actual_orders"])
    # Solvers store reflectance in an array with three axes
    # Rs[wavelength_index, theta_index, phi_index]
    # Currently model only has one of each
    fmmax_Rs.append(float(results["fmmax"]["Rs"][0,0,0]))

    s4_orders.append(results["s4"]["actual_orders"])
    s4_Rs.append(float(results["s4"]["Rs"][0,0,0]))

    metarcwa_orders.append(results["metarcwa"]["actual_orders"])
    metarcwa_Rs.append(float(results["metarcwa"]["Rs"][0,0,0]))

# Plot results
plt.figure(figsize=(8,5))

plt.plot(s4_orders,s4_Rs, marker = "o", label = "S4")
plt.plot(fmmax_orders, fmmax_Rs, marker="x", label="FMMax")
plt.plot(metarcwa_orders,metarcwa_Rs,marker="^",label="MetaRCWA")
plt.xlabel(r"Actual number of retained Fourier harmonics, $N_h$")
plt.ylabel(r"$R_s$")
plt.title("Total $s$-polarized reflectance vs retained basis size")
plt.grid(True)
plt.legend()
plt.tight_layout()
# Locate the project directory from this script's location.
project_directory = Path(__file__).resolve().parents[1]

output_directory = project_directory / "outputs" / "square"
output_directory.mkdir(parents=True, exist_ok=True)
csv_path = output_directory / "square_convergence.csv"

columns = [
    "backend",
    "m",
    "n",
    "requested_orders",
    "actual_orders",
    "Rs",
    "Rp",
    "Ts",
    "Tp"
]

with csv_path.open("w",newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"Saved CSV to {csv_path}")

plt.savefig(
    output_directory / "square_convergence.png",
    dpi=200,
)
plt.show()