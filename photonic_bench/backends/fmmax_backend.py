"""
Translate a MetaRCWA benchmark into FMMax operations.
"""
from photonic_bench.model.result import SimulationResult
from photonic_bench.model.config import BenchmarkConfig

def run_fmmax(benchmark: BenchmarkConfig) -> SimulationResult:
    """
    Run one benchmark using FMMax.

    The function first resolves the MetaRCWA model onto a numerical 
    grid. It then translates that numerical description into FMMax operations. 

    Parameters:
    -----------
    benchmark:
        Physical model, common numerical settings and FMMax settings.
    
    Returns:
    -----------
    SimulationResult
        s/TE and p/TM reflectance and runtime in the common result format
    """

    # Evaluates the materials and will rasterise the patterned layer onto the
    # requested grid
    model_spec = benchmark.model.spec(
        nx=benchmark.numerics.nx,
        ny=benchmark.numerics.ny,
    )

    # TODO: 
    # 1. Read the lattice and source from model_spec
    # 2. Create the FMMaz Fourier Expansion
    # 3. Translate and solve each layer
    # 4. Construct the scattering matrix
    # 5. Extract the s/TE and p/TM relfectance
    # 6.Measure runtime

    # Once the calculation is implemented
    # we return the simulation result

    # return SimulationResult(
    #     solver="fmmax",
    #     reflectance_te=reflectance_te,
    #     reflectance_tm=reflectance_tm,
    #     runtime_seconds=runtime_seconds,
    # )