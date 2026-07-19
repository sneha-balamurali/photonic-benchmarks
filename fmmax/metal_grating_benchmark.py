"""Originally based on FMMax's metal_grating.py example.

Copyright (c) Meta Platforms, Inc. and affiliates.

Modified by Sneha Balamurali for benchmarking different RCWA solvers.
"""

import csv
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt

import itertools
from typing import Tuple

import jax.numpy as jnp

from fmmax import basis, fmm, scattering, utils

NUM_TERMS_SWEEP = (9, 25, 49, 81, 121, 169, 225, 289, 361, 441, 529, 625, 729, 841)


def simulate_grating(
    permittivity_ambient: complex = 1.0 + 0.0j,
    permittivity_planarization: complex = 2.25 + 0.0j,
    permittivity_substrate: complex = -7.632 + 0.731j,
    wavelength_nm: float = 500.0,
    pitch_nm: float = 180.0,
    grating_width_nm: float = 60.0,
    grating_thickness_nm: float = 80.0,
    planarization_thickness_nm: float = 20.0,
    resolution_nm: float = 1.0,
    approximate_num_terms: int = 20,
    truncation: basis.Truncation = basis.Truncation.CIRCULAR,
    formulation: fmm.Formulation = fmm.Formulation.FFT,
) -> Tuple[int, complex, complex]:
    """Computes the TE- and TM-polarized reflection from a 1D stripe grating.

    Args:
        permittivity_ambient: The permittivity of the ambient.
        permittivity_planarization: The permittivity of media encapsulating grating.
        permittivity_substrate: The permittivity of the substrate below the grating,
            and the grating itself.
        wavelength_nm: The excitation wavelength, in nanometers.
        pitch_nm: The grating pitch, in nanometers.
        grating_width_nm: The width of the lines comprising the grating.
        grating_thickness_nm: The height of the grating.
        planarization_thickness_nm: The thickness of the planarization layer above
            the grating.
        resolution_nm: The rasterization resolution for patterned layers.
        approximate_num_terms: The approximate number of terms used in the plane
            wave expansion of the fields.
        truncation: Determines the truncation of the expansion.
        formulation: Specifies the formulation to be used.

    Returns:
        The number of terms in the expansion, and the reflection coefficients for TE-
        and TM-polarization.
    """
    x_nm, _ = jnp.meshgrid(
        jnp.arange(-pitch_nm / 2, pitch_nm / 2, resolution_nm),
        jnp.arange(-pitch_nm / 2, pitch_nm / 2, resolution_nm),
        indexing="ij",
    )
    density = (jnp.abs(x_nm) <= grating_width_nm / 2).astype(float)

    permittivities = [
        jnp.asarray([[permittivity_ambient]]),
        jnp.asarray([[permittivity_planarization]]),
        utils.interpolate_permittivity(
            permittivity_solid=jnp.asarray(permittivity_substrate),
            permittivity_void=jnp.asarray(permittivity_planarization),
            density=density,
        ),
        jnp.asarray([[permittivity_substrate]]),
    ]
    thicknesses = [0, planarization_thickness_nm, grating_thickness_nm, 0]

    in_plane_wavevector = jnp.asarray([0.0, 0.0])
    primitive_lattice_vectors = basis.LatticeVectors(
        u=jnp.asarray([pitch_nm, 0.0]), v=jnp.asarray([0.0, pitch_nm])
    )
    expansion = basis.generate_expansion(
        primitive_lattice_vectors=primitive_lattice_vectors,
        approximate_num_terms=approximate_num_terms,
        truncation=truncation,
    )
    layer_solve_results = [
        fmm.eigensolve_isotropic_media(
            wavelength=jnp.asarray(wavelength_nm),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            permittivity=p,
            expansion=expansion,
            formulation=formulation,
        )
        for p in permittivities
    ]
    s_matrix = scattering.stack_s_matrix(
        layer_solve_results=layer_solve_results,
        layer_thicknesses=[jnp.asarray(t) for t in thicknesses],
    )

    r_te = s_matrix.s21[0, 0]
    r_tm = s_matrix.s21[expansion.num_terms, expansion.num_terms]
    return expansion.num_terms, complex(r_te), complex(r_tm)


def convergence_study(
    approximate_num_terms: Tuple[int, ...] = NUM_TERMS_SWEEP,
    truncations: Tuple[basis.Truncation, ...] = (
        basis.Truncation.CIRCULAR,
        basis.Truncation.PARALLELOGRAMIC,
    ),
    fmm_formulations: Tuple[fmm.Formulation, ...] = (
        fmm.Formulation.FFT,
        fmm.Formulation.JONES_DIRECT,
        fmm.Formulation.JONES,
        fmm.Formulation.NORMAL,
        fmm.Formulation.POL,
    ),
) -> Tuple[Tuple[fmm.Formulation, basis.Truncation, int, complex, complex], ...]:
    """Sweeps over number of terms and fmm formulations to study convergence."""
    results = []
    for formulation, truncation, n in itertools.product(
        fmm_formulations,
        truncations,
        approximate_num_terms,
    ):
        
        start_time = time.perf_counter()

        num_terms, r_te, r_tm = simulate_grating(
            approximate_num_terms=n,
            truncation=truncation,
            formulation=formulation,
        )

        runtime_seconds = time.perf_counter() - start_time

        results.append((formulation, truncation, num_terms, r_te, r_tm, runtime_seconds))
        print(
            f"{formulation.value}/{truncation.value}/n={num_terms}: "
            f"r_te={complex(r_te):.3f}, r_tm={complex(r_tm):.3f}, runtime={runtime_seconds:.3f}s"
        )

    results_directory = Path("results")
    results_directory.mkdir(exist_ok=True)

    output_path = results_directory / "metal_grating_benchmark.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "formulation",
                "truncation",
                "num_terms",
                "r_te_real",
                "r_te_imag",
                "r_te_magnitude",
                "r_tm_real",
                "r_tm_imag",
                "r_tm_magnitude",
                "runtime_seconds",
            ]
        )
        for (formulation, truncation, num_terms, r_te, r_tm, runtime_seconds) in results:
            writer.writerow(
                [
                    formulation.value,
                    truncation.value,
                    num_terms,
                    r_te.real,
                    r_te.imag,
                    abs(r_te),
                    r_tm.real,
                    r_tm.imag,
                    abs(r_tm),
                    runtime_seconds,
                ]
            )
    print(f"\nSaved results to {output_path}")
    return tuple(results)

def plot_results(results):
    """Plots convergence and runtime for the FFT formulation."""
    # Reflection Coefficient Plot

    for truncation in(basis.Truncation.CIRCULAR, basis.Truncation.PARALLELOGRAMIC):
        terms = []
        te = []
        tm = []

        for result in results:
            if result[1] == truncation:
                terms.append(result[2])
                te.append(abs(result[3]))
                tm.append(abs(result[4]))
        
        plt.plot(terms, te, marker='o', label=f'TE - {truncation.value}')

        plt.plot(terms, tm, marker='x', label=f'TM - {truncation.value}')

    plt.xlabel("Number of Terms")
    plt.ylabel("Reflection Coefficient Magnitude")
    plt.title("Convergence of FFT Formulation forMetal Grating Simulation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/fft_convergence_plot.png", dpi=200)
    plt.close()

    # Runtime Plot

    for truncation in (basis.Truncation.CIRCULAR, basis.Truncation.PARALLELOGRAMIC):
        terms = []
        runtimes = []

        for result in results:
            if result[1] == truncation:
                terms.append(result[2])
                runtimes.append(result[5])

        plt.plot(terms, runtimes, marker='o', label=f'Runtime - {truncation.value}')

    plt.xlabel("Number of Terms")
    plt.ylabel("Runtime (seconds)")
    plt.title("FFT Runtime of Metal Grating Simulation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/fft_runtime_plot.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    results = convergence_study(fmm_formulations=(fmm.Formulation.FFT,))
    plot_results(results)