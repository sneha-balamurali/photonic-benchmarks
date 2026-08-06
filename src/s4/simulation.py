from dataclasses import dataclass
from typing import Any

import S4
import torch
from metarcwa import Model

from src.config import Config
from src.s4.config import S4Config

def tensor_to_tuple(value:torch.Tensor) -> tuple[float,float]:
    """Convert a two-component PyTorch tensor into an ordinary
    Python tuple.

    MetaRCWA stores each lattice vector as a PyTorch tensor 
    with shape (2,) but S4's python interface expects 
    something like ((a1_x,a1_y), (a2_x,a2y))

    Gradient tracking and GPU storage are not meaningful to S4, 
    so the tensor is detached and moved to CPU before extracting
    its values.
    """

    flattened = value.detach().cpu().reshape(-1)

    return (float(flattened[0]),float(flattened(1)))

@dataclass
class PreparedS4Model:
    """Initial S4 objects prepared from a MetaRCWA model.

    This first stage creates only the S4 simulation, lattice
    and Fourier settings. Materials, layer, geometry and illumination
    are added in later stages.

    Attributes
    ----------
    config:
        Common numerical settings translated into S4 compatible values.
    model_spec:
        MetRCWA's resolved model description. It contains evaluated
        materials, layers, lattice vectors, wavelengths and wavevectors.
    lattice_vectors:
        Two dimension lattice vectors converted into ordinary Python
        numbers for S4
    simulation:
        S4 simulation object configured with the lattice, requested 
        basis size and lattice truncation rule
    """

    config: S4Config
    model_spec: Any
    lattice_vectors: tuple[
        tuple[float,float],
        tuple[float,float]
    ]
    simulation: Any

    @classmethod
    def from_model(
        cls, 
        model: Model,
        config: Config,
    ) -> "PreparedS4Model":
    """Translate a MetaRCWA model into S4 simulation objects.

    This methods:
    
    - translates common numerical configuration to S4 compatible
    - resolves and rasterises the MetaRCWA model
    - creates the lattice vectors for S4
    - creates the simulation configured with the lattice, requested
    basis size and lattice truncation rule.
    """

    cfg = S4Config.from_config(config)

    # The original model contains higher-level description
    # such as shapes, layers and etc. Calling spec() evaluates
    # those descriptions such as evaluating permittivity at
    # requested wavelengths, nx, ny and etc.

    model_spec = model.spec(nx=cfg.nx,
                            ny=cfg.ny)

    # MetaRCWA stores lattice vectors as PyTorch tenors
    # S4 expects something of the form:
    # ((a1_x,a1_y), (a2_x,a2y))
    # A nested tuple with two 2D lattice vectors
    lattice_vectors = (
        tensor_to_tuple(model.spec.a1),
        tensor_to_tuple(model.spec.a2)
    )

    simulation = S4.New(
        Lattice=lattice_vectors,
        NumBasis=s4_config.requested_num_basis
    )

    simulation.SetOptions(LatticeTruncation= 
                        s4_config.latice_truncation)

    return cls(
        config=s4_config,
        model_spec=model_spec,
        lattice_vectors=lattice_vectors,
        simulation=simulation
    )