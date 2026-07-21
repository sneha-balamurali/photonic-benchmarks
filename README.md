# Photonic Benchmarks

This repository contains benchmarking implementations for Rigorous Coupled Wave Analysis (RCWA) solvers.

The aim is to compare different RCWA implementations using identical physical structure and simulation parameters.

**Repository Provides:**

- benchmark geometries
- convergence studies 
- runtime comparisons
- accuracy comparisons
- documentation


## Current benchmark implementations:

- [FMMax (JAX)](https://github.com/facebookresearch/fmmax)
- [S⁴](https://web.stanford.edu/group/fan/S4/)

## Repository Structure

**FMMax:**
- Metal Grating:
    - [metal_grating_benchmark.py](fmmax/metal_grating_benchmark.py)
        - Main FMMax benchmark implementation
        - Performs convergence studies for the metal grating benchmark and records reflection coefficients
        - Serves as the reference implementation that the S4 version reproduces
    - [metal_grating_debug.py](fmmax/metal_grating_debug.py)
        - Debugging version of the FMMax benchmark
        - Prints intermediate quantities and checks for NaN values to help diagnose numerical issues
        - Used for validating modifications before incorporating into main benchmark

**$S^4$:**
- Metal Grating:
    - [metal_grating.lua](s4/metal_grating.lua)
        - S4 implementation of the FMMax metal grating benchmark
        - Reproduces the benchmark geometry, material parameters, and simulation settings using the $S^4$ Lua interface
        - Will be used to compare reflection coefficients, convergence behaviour and runtime with the FMMax implementation
    - [plot_metal_grating.py](s4/plot_metal_grating.py)
        - Reads the benchmark CSV with the results from metal_grating.lua and generates convergence and runtime plots.
    - [metal_grating_doc.md](s4/metal_grating_doc.md)
        - Documentation describing the S4 benchmark implementation
    - [results/](s4/results/)
        - Stores CSV benchmark outputs and generated figures