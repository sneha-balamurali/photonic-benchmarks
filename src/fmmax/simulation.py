
import jax
import torch
from typing import Tuple
from typing import Any

# JAX disables float64 by default. Enabling it allows the 
# adapter to run with float64 precision.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from fmmax import basis, utils
from src.fmmax.config import FMMaxConfig
from metarcwa import Model
from src.config import Config
from metarcwa.model.layer import HomogeneousLayer, PatternedLayer


def torch_to_jax(value: torch.Tensor, 
                dtype:type,) -> Any:
    """Converts a MetaRCWA torch tensor into a JAX array.
    
    MetaRCWA uses PyTorch tensors, whereas FMMax uses JAX arrays.
    This conversion is necessary to run the FMMax solver on the same model.
    
    This function uses detach() to remove PyTorch's gradient tracking
    because a tensor connected to a computation graph cannot be converted 
    to a NumPy array. It then passes the tensor into CPU memory if it is
    currently on a GPU. This is because NumPy does not support GPU tensors.

    It is then converted to a NumPy array because both PyTorch and JAX understand 
    NumPy's in-memory array format. The NumPy conversion gives a predictable
    CPU array that can be passed into JAX's jnp.asarray() function to create a JAX array.

    Something I want to explore later is DLPack, which allows JAX to use 
    PyTorch tensors directly but for simple conversion, this is a good first step.

    Parameters
    ----------
    value: 
        Pytorch tensor to be converted into a JAX array.
    dtype:
        The desired data type for the JAX array.
    
    Returns
    -------
    Any:
        JAX array with the same data as the input tensor,
        but in the specified dtype.
    """ 

    numpy_value = value.detach().cpu().numpy()

    jax_array = jnp.asarray(numpy_value, dtype=dtype)

    return jax_array

def prepare_fmmax_model(model:Model,
                        config:Config) -> Tuple[Any, Any, Any]:
    """ Prepares the FMMax model from the MetaRCWA model and config.

    Just a first translation stage. It:

    - Converts the common Config into FMMax-specific numerical settings.
    - Resolves the MetaRCWA model at the requested resolution
    - Converts the MetaRCWA lattice vectors from Pytorch tensors to JAX arrays.
    - Generates the FMMax Fourier Expansion.

    It does not yet translate layer permittivities, solve layer eigenmodes,
    build the scattering matrix or calculate reflection or transmission coefficinet.
    
    Parameters
    ----------
    mode:
        Solver-independent MetaRCWA physical model.
    config:
        Common configuration object with numerical settings
        shared between solver adapters.
    
    Returns
    -------
    tuple:
        Tuple containing the resolved model spec,
        the JAX arrays for the lattice vectors a1 and a2, and the
        FMMax Fourier expansion.

    - model_spec: The resolved MetaRCWA model specification.
    - lattice_vectors: The FMMax lattice vectors as JAX arrays.
    - expansion: The Fourier basis selected by FMMax.
    """

    cfg = FMMaxConfig.from_config(config)

    # Resolving the model evaluates the materials at the source
    # wavelengtyh and rasterises the pattern into a 2D array onto
    # the nx by ny grid.
    model_spec = model.spec(
        nx=cfg.nx,
        ny=cfg.ny,
    )

    # MetaRCWA calls its lattice vectors a1 and a2.
    # FMMax calls them u and v. They are the same vectors.
    lattice_vectors = basis.LatticeVectors(
        u = torch_to_jax(model_spec.a1,dtype=cfg.real_dtype),
        v = torch_to_jax(model_spec.a2,dtype=cfg.real_dtype),
    )

    # Convert the model.spec wavelength into a JAX array.
    wavelength = torch_to_jax(model_spec.wavelength,dtype=cfg.real_dtype)

    # Convert the incidence and transmission 
    # permittivities to Jax arrays
    incidence_eps = torch_to_jax(model_spec.incidence.eps,dtype=cfg.complex_dtype)
    transmission_eps = torch_to_jax(model_spec.transmission.eps,dtype=cfg.complex_dtype)
    
    # Convert the layer permittivities and thicknesses to JAX arrays.
    layer_permittivities = []
    layer_thicknesses = []

    for layer in model_spec.layers:
        thickness = torch_to_jax(layer.thickness,dtype=cfg.real_dtype)
        layer_thicknesses.append(thickness)

        if isinstance(layer, HomogeneousLayer):
            permittivity = torch_to_jax(layer.eps, dtype=cfg.complex_dtype)
            layer_permittivities.append(permittivity)
            elif isinstance(layer, PatternedLayer):
                       solid_eps = torch_to_jax(layer.solid.eps, dtype=cfg.complex_dtype)
                       void_eps = torch_to_jax(layer.void_eps, dtype=cfg.complex_dtype)    

    pass
    
    # FMMax creates the actual reciprocal-space Fourier basis.
    # Its final number of terms may differ slightly from
    # approximate_num_terms to maintain symmetry in the expansion.
    expansion = basis.generate_expansion(
        primitive_lattice_vectors=lattice_vectors,
        approximate_num_terms=cfg.approximate_num_terms,
        truncation=cfg.truncation,  
    )

    return (model_spec, 
            lattice_vectors, 
            expansion,
            wavelength,
            incidence_eps,
            transmission_eps,
            layer_permittivities,
            layer_thicknesses)

# def run_fmmax(model: Model, config: Config) -> Tuple[torch.Tensor, torch.Tensor,
#                                                         torch.Tensor, torch.Tensor]:
#     """ Runs fmmax computation. Need to declare 32/64 bits and device before running. """
#     # Translate the config
#     cfg = FMMaxConfig.from_config(config)
#     ...
    
#     # Replace for real computation
#     Rs = torch.zeros()
#     Rp = torch.zeros()
#     Ts = torch.zeros()
#     Tp = torch.zeros()
#     return Rs, Rp, Ts, Tp
