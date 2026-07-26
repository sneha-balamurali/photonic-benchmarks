# Photonic Benchmarks

This repository contains benchmarking implementations for Rigorous Coupled Wave Analysis (RCWA) solvers.

The aim is to represent the same physical structures in different RCWA solvers and compare their reflection results, convergence behaviour, and solver-specific runtimes.

**Repository Provides:**

- benchmark geometries
- convergence studies 
- runtime comparisons
- accuracy comparisons
- documentation


## Solvers:

- [FMMax (JAX)](https://github.com/facebookresearch/fmmax)
- [S⁴](https://web.stanford.edu/group/fan/S4/)

## Repository Structure

**FMMax:**
- Metal Grating:
    - [metal_grating_benchmark.py](fmmax/metal_grating_benchmark.py)
        - Main FMMax benchmark implementation
        - Performs convergence studies for the metal grating benchmark and records reflection coefficients
        - Defines the original metal-grating geometry reproduced by the S4 implementation
        - a s/TE wavelength sweep from 400-700nm in 5nm steps at 841 harmonic terms
    - [metal_grating_debug.py](fmmax/metal_grating_debug.py)
        - Debugging version of the FMMax benchmark
        - Prints intermediate quantities and checks for NaN values to help diagnose numerical issues
        - Used for validating modifications before incorporating into main benchmark

**$S^4$:**
- Metal Grating:
    - [metal_grating.lua](s4/metal_grating.lua)
        - S4 implementation of the FMMax metal grating benchmark
        - Reproduces the benchmark geometry, material parameters, and simulation settings using the $S^4$ Lua interface
        - Runs convergence or wavelength studies selected through the study command-line variable. It reports complex specular reflection coefficients, reflectance, and CPU time.
    - [plot_metal_grating.py](s4/plot_metal_grating.py)
        - Reads the benchmark CSV with the results from metal_grating.lua and generates convergence and runtime plots.
    - [metal_grating_doc.md](s4/metal_grating_doc.md)
        - Documentation describing the S4 benchmark implementation
    - [results/](s4/results/)
        - Stores CSV benchmark outputs and generated figures

**Cross-solver comparisons:**

- [compare_metal_grating.py](Comparison/compare_metal_grating.py)
  - Compares FMMax FFT with parallelogramic truncation against S4 FFT.
  - Produces separate TE/s and TM/p convergence plots.

- [compare_metal_grating_wavelength_sweep.py](Comparison/compare_metal_grating_wavelength_sweep.py)
  - Compares the FMMax TE and S4 s-polarized wavelength sweeps.

- [Comparison results](Comparison/results/)
  - [compare_te_convergence_1.png](Comparison/results/compare_te_convergence_1.png)
  - [compare_tm_convergence_1.png](Comparison/results/compare_tm_convergence_1.png)
  - [compare_te_wavelength_sweep.png](Comparison/results/compare_te_wavelength_sweep.png)

**Running the benchmarks:**
- [Basic run commands for the scripts](running_benchmarks.md)

