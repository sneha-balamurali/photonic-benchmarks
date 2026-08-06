from dataclasses import dataclass

from src.config import Config

# The common Config uses simple lowercase strings because they are 
# convenient to store in YAML. S4 expects these specific capitalised
# option names. 
_TRUNCATION_MAP= {
    "circular": "Circular",
    "rectangular": "Parallelogramic"
}

@dataclass
class S4Config:
    """Numerical settings translated into S4 compatible format.

    Attributes
    ----------
    nx,ny: 
        Grid sizes passed to model.spec(nx,ny). MetaRCWA uses these values to
        rasterise patterned layers. S4 uses analytical geometry commands like 
        `S.SetRegionRectangle()` so the S4 solver does not use 
        this grid.
    requested_num_basis:
        Maximum number of in-plane Fourier orders requested from S4. S4 may 
        retain slightly different number depending on the lattice and truncation
        rule. 
    lattice_truncation:
        S4's name for the Fourier-order truncation rule. It is either `Circular`
        or `Parallelogramic`.
    """

    nx: int
    ny: int
    requested_num_basis: int
    lattice_truncation: str
    
    @classmethod
    def from_config(cls, config: Config) -> "S4Config":
        """ Translate the common Config into S4 settings"""

        if config.truncation not in _TRUNCATION_MAP:
            raise ValueError(
                f"Unsupported truncation {config.truncation!r}"
                f"Choose one of: {','.join(_TRUNCATION_MAP)}"
            )

        # The common Config provides separate maximum indices m and n, whereas
        # S4 accepts one requested basis size. Use the size of the complete
        # two-dimensional order grid as the initial NumBasis target.
        #
        # S4 will choose the actual order pairs later using its own lattice and
        # truncation rule.
        requested_num_basis = (
            (2 * config.m + 1) * (2 * config.n + 1)
        )

        return cls(
            nx = config.nx,
            ny=config.ny,
            requested_num_basis=requested_num_basis,
            lattice_truncation=_TRUNCATION_MAP[config.truncation]
        )    