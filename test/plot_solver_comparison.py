"""Create comparison plot for MetaRCWA, FMMax and S4"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from dispertorch import ConstantEps

from examples.metarcwa.square_particle import build_model
from src.config import Config
from src.fmmax.simulation import run_fmmax
from src.metarcwa.simulation import run_metarcwa
from src.s4.simulation import run_s4

def create_config() -> Config:
    """Creare a simple numerifal config for all three solvers."""

    return Config(
        dtype="float64",
        device="cpu",
        nx=32,
        ny=32,
        m=2,
        n=2,
        truncation="circular"
    )

def create_model(
    wavelengths_nm: np.ndarray,
    theta_rad: np.ndarray,
    phi_rad: np.ndarray
):
    """Create the same square particle model for each solver.

    Wavelength, theta and phi are independent sweep axes. 
    MetaRCWA, FMMax and S4 will calculate every combination of 
    these values. 

    """

    return build_model(
        particle_side_length_nm=torch.tensor(
            60.0,
            dtype=torch.float64,
        ),
        planarization_layer_thickness_nm=torch.tensor(
            20.0,
            dtype=torch.float64,
        ),
        patterned_layer_thickness_nm=torch.tensor(
            80.0,
            dtype=torch.float64,
        ),
        lattice_period_nm=torch.tensor(
            180.0,
            dtype=torch.float64,
        ),
        wavelength_nm=torch.as_tensor(
            wavelengths_nm,
            dtype=torch.float64,
        ),
        theta_rad=torch.as_tensor(
            theta_rad,
            dtype=torch.float64,
        ),
        phi_rad=torch.as_tensor(
            phi_rad,
            dtype=torch.float64,
        ),
        incidence_perm_model=ConstantEps(1.0),
        planarization_perm_model=ConstantEps(2.25),
        particle_perm_model=ConstantEps(
            eps_re=-7.632,
            eps_im=0.731,
        ),
        pattern_background_perm_model=ConstantEps(2.25),
        transmission_perm_model=ConstantEps(
            eps_re=-7.632,
            eps_im=0.731,
        ),
    )

def to_numpy(value) -> np.ndarray:
    """Convert a solver result into a flat NumPy array for plotting.

    MetaRCWA and S4 return PyTorch tensors. FMMax returns JAX arrays. 
    Matplotlib works with NumPy arrays so this function provides a way
    to convert them to numpy arrays for both cases."""

    if isinstance(value, torch.Tensor):
        return(value
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

    return np.asarray(value).reshape(-1)

def run_all_solvers(
    model,
    config:Config
) -> dict[str, tuple[np.ndarray,...]]:

    """Run all three solvers and convert their outputs for plotting."""

    solver_functions = {
        "MetaRCWA": run_metarcwa,
        "FMMax": run_fmmax,
        "S4": run_s4
    }

    results = {}

    for solver_name, solver_function in solver_functions.items():
        print(f"Running{solver_name}...")

        Rs, Rp, Ts, Tp = solver_function(
            model = model,
            config = config
        )

        results[solver_name] = (
            to_numpy(Rs),
            to_numpy(Rp),
            to_numpy(Ts),
            to_numpy(Tp)
        )
    return results

def main() -> None:
    """Run a preliminary normal-incidence wavelength comparison."""

    config = create_config()

    wavelengths_nm = np.linspace(
        450,
        650,
        5
    )

    model = create_model(
        wavelengths_nm = wavelengths_nm,
        theta_rad=np.array([0.0]),
        phi_rad=np.array([0.0])
    )

    results = run_all_solvers(
        model=model,
        config=config
    )

    quantity_names = (
        "Rs",
        "Rp",
        "Ts",
        "Tp"
    )

    solver_styles = {
        "MetaRCWA": {
            "marker": "o",
            "linestyle": "-"
        },

        "FMMax": {
            "marker": "x",
            "linestyle": "--",
        },

        "S4": {
            "marker": "^",
            "linestyle":"-."
        }
    }

    # Creates four plots, with shared legend and title
    # 2 rows and columms to give 4 subplots for Rs, Rp, Ts, Tp
    figure, axes = plt.subplots(
        nrows=2,
        ncols = 2,
        figsize=(11,8),
        # all four panels share the same x axis range
        sharex=True
    )

    # axes.flat turns the 2x2 array of axes into a sequence:
    # top left, top right, bottom left, bottom right
    for quantity_index, quantity_name in enumerate(quantity_names):
        axis = axes.flat[quantity_index]

        for solver_name, solver_values in results.items():
            y_values = solver_values[quantity_index]

            axis.plot(
                wavelengths_nm,
                y_values,
                label=solver_name,
                linewidth=1.8,
                markersize=5,
                **solver_styles[solver_name]
            )

        axis.set_title(quantity_name)
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel("Power fraction")

        # Negative results may reveal a numerical problem
        # So not setting the lower limit to zero during this
        # validation
        axis.axhline(
            0.0,
            color="grey",
            linewidth=0.8
        )

        axis.grid(
            visible=True,
            alpha=0.3
        )

    # Use one shared legend instead of repeating it in every panel
    legend_handles, legend_labels = axes[0,0].get_legend_handles_labels()

    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5,0.93),
        ncol=3
    )

    figure.suptitle(
        "Preliminary RCWA solver comparison for nondispersive square particle\n"
        "with illumination at normal incidence to surface."
    )

    figure.tight_layout(
        rect=(0.0,0.0,1.0,0.95)
    )

    output_path = Path(
        "results/preliminary_solver_comparison.png"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    print(f"Saved plot to: {output_path.resolve()}")

    plt.close(figure)

if __name__ == "__main__":
    main()
