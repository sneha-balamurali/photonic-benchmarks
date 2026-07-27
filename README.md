# Photonic Benchmarks

## Project Goal

![workflow](image/metal_grating.png)

The goal of the internship is to develop a python based benchmarking framework for comparing RCWA electromagnetic solvers.

A user will be able to define a periodic layered photonic structure and its simulation settings in a YAML file, Python then loads it into a structured configuration object and the appropriate backend translates the configuration into FMMax or S4 commands. 

S4 and FMMax will return results in a standardised format so that their outputs can be compared without writing seperate comparison logic for every structure. 

The framework will initially be demonstrated using a 1D metal grating/2D square particle. 

The project will also have reusable functions for convergence, and nondispersive wavelength studies and etc. 

The main framework will be completed by the end of August. The remaining time in September will be reserved for debugging, numerical investigation, documentation and any required report.

## Project Layout

Photonic bench:
- `configs/`: Stores values for each simulation
    - `metal_grating.yaml`
    - `square_particle.yaml`
- `model/` : Defines how the physical problem and numerical settings and results are represented in Python e.g.: 
    - `config.py`
        - defines the configuration dataclasses
        - converts the configuration to and from dictonaries
        - contains the common physical settings and solver-specific numerical settings
    - `result.py`
        - defines the standard result format returned by FMMax and S4
- `backends/`: Translates the common configuration into solver ready format
    - `s4_backend.py`
        - Creates the s4 simulation
        - Creates the materials, lattice, layers and patterns
        - Applies the S4 numerical options
        - Runs the simulation
        - Extracts and returns the results
    - `fmmax_backend.py`  
        - Creates the FMMax lattice and geometry
        - Generates the Fourier expansions
        - Solves the layers and constructs the scattering matrix
        - Extracts and returns the requested results
- `studies/`: While the backend performs one simulation, a study can perform several simulations in a controlled series. 
    - `convergence.py`
        - Changes basis while physical problem is fixed
    - `wavelength_sweep.py`
        - Changes wavelength while holding everything else fixed. 
- `validation/`: To check whether the results returned appear reliable
    - `failures.py`
        - Checks for NaNs, infinite values
    - `discrepancy.py`
        - Compares FMMax and S4 results
- `reports/`: Turns results into readable figures and summaries
    - `plots.py`
        - TE convergence plots, TM convergence plots, wavelength plots, runtime plots and etc
- `results/`: Contains generated CSV files, plots and other data
- `docs/`: Usage instructions and documents any issues found 
- `run_benchmark.py`: The script you run from terminal to load YAML, create the Python configuration object and run your simulation and print or save results

## Week 1

### Goal: Implement the smallest complete YAML to S4 simulation route

### Tasks:
- Create a new development branch on Git
- Check the parameters currently coded into the S4 Python files
- Define the first S4 config dataclass
- Implement `to_dict()` and `from_dict()`
- Create a matching metal grating or square particle YAML file
- Include the requried S4 formulations, precision, and options settings
- Implement the S4 backend for one simulation
- Return a structured S4 result

## Week 2

### Goal: Implement the smallest complete YAML to FMMAx simulation route

### Tasks:
- Identify the FMMax specific settings
- Implement the FMMax YAML and configuration route
- Implement the FMMax backend 
- Define the results and update S4 backend if needed to also return this result format
- Reproduce metal grating/square particle test
- Compare results with previous results 

## Week 3

### Goal: Add studies and another geometry

### Tasks:
- Implement the convergence study
- Implement the wavelength study
- Add another geometry to FMMax and S4

## Week 4:

### Goal: Numerical Validation 

### Tasks:
- Add failure/non-finite value reporting
- Add solver-discrepancy calculations to compare FMMax and S4 results
- Investigate TM fluctuations
- Record the actual basis used by each solver

## Week 5: 

### Goal: Producable a useable framework

### Tasks: 
- Clean package layout
- Document installation and etc
- Record any issues found
