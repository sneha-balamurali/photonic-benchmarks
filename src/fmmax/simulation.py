from dataclasses import dataclass
from pathlib import Path

import jax

# Enable double precision before creating JAX arrays
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import yaml

from fmmax import basis, fields, fmm, scattering, utils

from dataclasses import replace
from pathlib import Path

import csv

@dataclass
class FMMaxModel:
    """Physical description of the physical structure

    Lengths are in nanometers, angles are stored in degrees
    and converted by each solver adapter as required.

    FMMax objects are prepared from one solver independent 
    MetaRCWA model.

    This dataclass stores the intermediate objects needed for a
    FMMax simulation. Class used because named fields are clearer
    than returning a long tuple whose meaning depends on the order
    of the position.

    Attributes
    ----------

    config:
        Numerical settings translated into FMMax compatible settings
    model_spec:
        MetaRCWA model after material evaluation and geometry rasterisation.
    lattice_vectors:
        Primitive lattice vectors u and v for the FMMax simulation


    """