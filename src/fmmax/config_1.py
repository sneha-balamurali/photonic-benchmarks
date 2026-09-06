from dataclasses import dataclass
from fmmax import basis, fmm

TRUNCATION_MAP = {
    "circular": basis.Truncation.CIRCULAR,
    "parallelogramic": basis.Truncation.PARALLELOGRAMIC,
    "rectangular": basis.Truncation.PARALLELOGRAMIC
}

FORMULATION_MAP = {
    "fft": fmm.Formulation.FFT,
    "jones": fmm.Formulation.JONES,
    "jones_direct": fmm.Formulation.JONES_DIRECT,
    "normal": fmm.Formulation.NORMAL,
    "pol": fmm.Formulation.POL
}

@dataclass
class FMMaxConfig:
    """Numerical settings used directly by FMMax.

    Attributes
    ---------
    nx, ny:
        Number of real-space samples used to represent the unit cell.
    approximate_num_terms:
        Approximate number of Fourier orders requested from FMMax.
    truncation:
        Rule used to select Fourier orders.
    formulation:
        Fourier-modal formulation used for patterned layers.
    """

    nx: int
    ny: int
    approximate_num_terms: int
    truncation: basis.Truncation
    formulation: fmm.Formulation

    def __post_init__(self) -> None:
        """Check that the numerical settings are valid."""

        if self.nx <= 0 or self.ny <=0:
            raise ValueError(
                "nx and ny must be positive"
            )
        if self.approximate_num_terms <=0:
            raise ValueError(
                "approximate_num_terms must be positive"
            )

    @classmethod
    def from_dict(cls, data:dict) -> "FMMaxConfig":
        """Create FMMax settings from YAML-derived dictionary data"""

        # Work on a copy so the original YAML dictionary is not changed
        data = dict(data)

        # strip() by default removes any whitespace at the beginning and end
        # lower() converts all letters to lowercase
        truncation_name = str(data["truncation"]).strip().lower()
        formulation_name = str(data["formulation"]).strip().lower()

        data["truncation"] = TRUNCATION_MAP[truncation_name]
        data["formulation"] = FORMULATION_MAP[formulation_name]

        return cls(**data)