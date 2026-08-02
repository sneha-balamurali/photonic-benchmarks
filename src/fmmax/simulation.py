
import jax
import torch
from typing import Tuple
from typing import Any

# JAX disables float64 by default. Enabling it allows the 
# adapter to run with float64 precision.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from fmmax import basis, utils, fmm, scattering
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
                        config:Config) -> Tuple[Any, ...]:
    """ Translate a MetaRCWA model into the initial FMMax simulation objects.

    This function:
    - translates the common numerical configuration
    - resolves and rasterises the MetaRCWA model
    - converts PyTorch tensors into JAX arrays
    - creates the FMMax lattice and Fourier expansion 
    - constructs homoegeneous and patterned permittivity arrays
    - solves the eigenmodes of every layer
    - combines the layers into a stack scattering matrix
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
    # MetaRCWA uses normalised in-planewave wavevector components kx0 and ky0, which are the
    # in-plane wavevector components divided by the free-space wavenumber k0 = 2 * pi / wavelength.
    # FMMax uses the actual in-plane wavevector components kx and ky.
    kx = torch_to_jax(model_spec.kx0, dtype=cfg.real_dtype) * (2 * jnp.pi / wavelength)
    ky = torch_to_jax(model_spec.ky0, dtype=cfg.real_dtype) * (2 * jnp.pi / wavelength)
    in_plane_wavevector = jnp.stack([kx,ky], axis=-1)

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
            permittivity = torch_to_jax(layer.medium.eps, dtype=cfg.complex_dtype)
        elif isinstance(layer, PatternedLayer):
                    solid_eps = torch_to_jax(layer.medium_solid.eps, dtype=cfg.complex_dtype)
                    void_eps = torch_to_jax(layer.medium_void.eps, dtype=cfg.complex_dtype)
                    pattern = torch_to_jax(layer.pattern, dtype=cfg.real_dtype)
                    permittivity = utils.interpolate_permittivity(permittivity_solid = solid_eps, 
                                                                    permittivity_void = void_eps, 
                                                                    density = pattern)
        layer_permittivities.append(permittivity)

    # Create a list of all permittivities, including incidence and transmission in order
    permittivities = [incidence_eps, *layer_permittivities, transmission_eps]

    # Create a list of all thicknesses, including incidence and transmission in order
    # The incidence and transmission media are semi-infinite, so they do not have finite
    # propagation thicknesses. FMMax represents them with zero thickness at the two ends
    # of the stack.
    zero_thickness = jnp.asarray(0.0, dtype=cfg.real_dtype)
    thicknesses = [zero_thickness, *layer_thicknesses, zero_thickness]

    # FMMax creates the actual reciprocal-space Fourier basis.
    # Its final number of terms may differ slightly from
    # approximate_num_terms to maintain symmetry in the expansion.
    expansion = basis.generate_expansion(
        primitive_lattice_vectors=lattice_vectors,
        approximate_num_terms=cfg.approximate_num_terms,
        truncation=cfg.truncation,  
    )

    # Solve the electromagnetic eigenmodes independetly in every medium.
    # and finite layer. The result describes how fields propagate through each layer.
    formulation = cfg.formulation
    layer_solve_results = [fmm.eigensolve_isotropic_media(wavelength = wavelength,
                                                        in_plane_wavevector = in_plane_wavevector,
                                                        primitive_lattice_vectors=lattice_vectors,
                                                        permittivity = permittivity,
                                                        expansion = expansion,
                                                        formulation = formulation) 
                                                        for permittivity in permittivities]
                                                        
    # Combine the individually solved layers into one scattering matrix for
    # the complete incidence to transmission stack.                                                    
    s_matrix = scattering.stack_s_matrix(layer_solve_results = layer_solve_result, 
                                        layer_thicknesses = thicknesses)

    return (model_spec, 
            lattice_vectors, 
            expansion,
            wavelength,
            in_plane_wavevector,
            incidence_eps,
            transmission_eps,
            permittivities,
            thicknesses,
            layer_solve_result,
            s_matrix)

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
