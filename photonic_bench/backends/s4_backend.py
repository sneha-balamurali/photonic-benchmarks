"""
Translate a MetaRCWA benchmark into a S4 operations

"""
from photonic_bench.model.result import SimulationResult
from photonic_bench.model.config import BenchmarkConfig

def run_s4(benchmark):
    """
    Run one benchmark using S4
    Parameters:
    -----------
    benchmark:
        physical model, common numerical settings based around MetaRCWA 
        and s4 specific settings
    
    Returns:
    --------
    SimulationResult
        s/TE and p/TM reflectance in the common result format
    """

    # Evaluate the MetaRCWA model before translation to S4
    model_spec = benchmark.model.spec(
        nx=benchmark.numerics.nx,
        ny=benchmark.numerics.ny,
    )

    # TODO
    #1. Create the S4 simulation
    #2. Create the S4 materials and lattice
    #3. Add the homogeneous and patterned layer
    #4. Apply the supported S4 settings
    #5. Set the source
    #6. Run S4 and extract s/TE and p/TM reflectance
    #7. Measure the runtime

    # Once the calculation is implemented, we return
    # the simulation result with the below

    # return SimulationResult(
    #     solver="s4",
    #     reflectance_te=reflectance_te,
    #     reflectance_tm=reflectance_tm,
    #     runtime_seconds=runtime_seconds,
    # )