"""Define the result format shared by all solver backends."""

from dataclasses import dataclass

@dataclass
class SimulationResult:
    """Small standard result returned by one solver run.

    Attributes:
    -----------
    solver:
        Name of the solver that produced the result
    reflectance_te:
        Reflected power for the TE/s polarised illumination
    reflectance_tm:
        Reflected power for TM/p polarised illumination
    runtime_seconds: 
        Time spent running the solver, when recorded
    """
    solver: str
    reflectance_te: float
    reflectance_tm: float
    runtime_seconds: float