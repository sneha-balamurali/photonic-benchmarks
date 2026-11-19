""" 
Translate the common Config into FMMax settings.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from fmmax import basis, fmm


from src.config import Config

DTYPE_MAP = {
    "float32": (
        jnp.float32,
        jnp.complex64
    ),
    "float64": (
        jnp.float64,
        jnp.complex128
    )
}

TRUNCATION_MAP = {
    "circular": basis.Truncation.CIRCULAR,
    "parallelogramic": basis.Truncation.PARALLELOGRAMIC
}

FORMULATION_MAP = {
    "fft": fmm.Formulation.FFT,
    "jones_direct": fmm.Formulation.JONES_DIRECT,
    "jones": fmm.Formulation.JONES,
    "normal": fmm.Formulation.NORMAL,
    "pol": fmm.Formulation.POL,
    "jones_direct_fourier": fmm.Formulation.JONES_DIRECT_FOURIER,
    "jones_fourier": fmm.Formulation.JONES_FOURIER,
    "normal_fourier": fmm.Formulation.NORMAL_FOURIER,
    "pol_fourier": fmm.Formulation.POL_FOURIER
}

@dataclass(frozen=True)
class FMMaxConfig:
    """
    Numerical settings translated for FMMax.
    """

    real_dtype: type
    complex_dtype: type
    device: str
    nx: int
    ny: int
    approximate_num_terms: int
    truncation: basis.Truncation
    formulation: fmm.Formulation

    @classmethod
    def from_config(cls,
                    config: Config,
                    formulation: str = "fft") -> "FMMaxConfig":
        """
        Create a configuration from a common Config.
        """
        # make formulation lower case
        formulation = formulation.lower()

        if formulation not in FORMULATION_MAP:
            raise ValueError(
                f"Unsupported FMMax formulation: {formulation}"
            )

        real_dtype, complex_dtype = DTYPE_MAP[config.dtype]

        # FMMax requests an approximate total number of terms
        approximate_num_terms = ((2 * config.m + 1) * ( 2*config.n + 1))

        truncation = TRUNCATION_MAP[config.truncation]
        formulation = FORMULATION_MAP[formulation]


        return cls(
            real_dtype=real_dtype,
            complex_dtype=complex_dtype,
            device=config.device,
            nx=config.nx,
            ny=config.ny,
            approximate_num_terms = approximate_num_terms,
            truncation=truncation,
            formulation=formulation
        )