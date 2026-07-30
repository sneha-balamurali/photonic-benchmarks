from dataclasses import dataclass

import jax.numpy as jnp
from fmmax import basis, fmm

from src.config import Config

_DTYPE_MAP = {
    "float32": jnp.float32,
    "float64": jnp.float64,
}

# The common Config uses strings because strings are easy to store in YAML.
# FMMax expects members of its Truncation enum instead. 
_TRUNCATION_MAP = {
    "circular": basis.Truncation.CIRCULAR,
    "rectangular": basis.Truncation.PARALLELOGRAMIC,
}


@dataclass
class FMMaxConfig:
    """Numerical settings translated into FMMax-compatible values.

    Unlike MetaRCWA, FMMax does not provide one Config class containing all
    its numerical settings. Its functions receive values such as basis size,
    truncation and formulation separately. This dataclass gathers those
    values into one object for the FMMax benchmark backend.

    Attributes
    ----------
    dtype:
        JAX floating-point type used when creating FMMax arrays.
    device:
        Requested execution device, stored for use by ``run_fmmax()``.
    nx, ny:
        Grid sizes used by ``model.spec(nx, ny)`` to rasterise patterned
        layers. These are not the number of Fourier terms.
    approximate_num_terms:
        Target number of Fourier terms requested from FMMax. FMMax may use a
        slightly different actual count to keep its expansion symmetric.
    truncation:
        Shape used to select Fourier terms in reciprocal space.
    formulation:
        FMMax method used for patterned layers. FFT is our initial baseline.
    """
    
    dtype: type
    device: str
    nx: int
    ny: int
    approximate_num_terms: int
    truncation: basis.Truncation
    formulation: fmm.Formulation = fmm.Formulation.FFT

    @classmethod
    def from_config(cls, config: Config) -> "FMMaxConfig":
        """Translate the common Config into FMMax-specific values.

        The common Config contains strings and MetaRCWA-style harmonic
        limits m and n. This method converts them into the values expected
        by FMMax.
        """

        # Check strings here so mistakes produce a clear error before FMMax
        # starts a more complicated numerical operation.
        if config.dtype not in _DTYPE_MAP:
            raise ValueError(
                f"Unsupported dtype {config.dtype!r}. "
                f"Choose one of: {', '.join(_DTYPE_MAP)}"
            )

        if config.truncation not in _TRUNCATION_MAP:
            raise ValueError(
                f"Unsupported truncation {config.truncation!r}. "
                f"Choose one of: {', '.join(_TRUNCATION_MAP)}"
            )

        # FMMax asks for one approximate total number of terms. 
        approximate_num_terms = (
            (2 * config.m + 1) * (2 * config.n + 1)
        )

        return cls(
            # Convert the common strings into objects JAX and FMMax expect.
            dtype=_DTYPE_MAP[config.dtype],
            device=config.device,
            nx=config.nx,
            ny=config.ny,
            approximate_num_terms=approximate_num_terms,
            truncation=_TRUNCATION_MAP[config.truncation],
        )
        
