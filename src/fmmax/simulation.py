import jax
import torch
from typing import Any
from dataclasses import dataclass

# JAX disables float64 by default. Enabling it 
# allows the adapter to run with float64 precision
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from fmmax import (basis,
                   utils,
                   fmm,
                   scattering,
                   fields)
from src.fmmax.config import FMMaxConfig
from src.config import Config
from metarcwa import Model
from metarcwa.model.layer import (
    HomogeneousLayer,
    PatternedLayer
)
from metarcwa.model.base import ModelSpec
from metarcwa.model.medium import IsotropicMediumSpec

def torch_to_jax(value: torch.Tensor,
                 dtype:type) -> Any:
    """
    Converts MetaRCWA torch tensor into a JAX
    array.

    Parameters
    ----------
    value:
        Pytorch tensor to be converted into a JAX
        array
    dtype:
        The desired data type for the JAX array
    
    Returns
    -------
    Any:
        JAX array with the same data as the input
        tensor, but in the specified dtype.
    """

    numpy_value = value.detach().cpu().numpy()

    jax_array = jnp.asarray(numpy_value,dtype=dtype)

    return jax_array

def prepare_uniform_permittivity(
        value: torch.Tensor,
        complex_dtype: type
) -> jnp.ndarray:
    """
    Converts a MetaRCWA uniform permittivity for batched FMMax calculations.

    In FMMax, the final two axes are interpreted as the sampled spatial
    dimensions of the unit cell. A spatially uniform medium has shape (1,1)
    in the final two dimensions because if permittivity is same everywhere,
    you only need one sample. If trailing dimensions are not (1,1), medium is
    interpreted as patterned. 
    """

    converted = torch_to_jax(value, dtype=complex_dtype)

    return converted[:, None, None, None, None]

@dataclass
class PreparedFMMaxModel:
    """
    Solver-specific intermediate objects, useful for
    inspecting a run.
    """

    config: FMMaxConfig
    model_spec: ModelSpec
    lattice_vectors: basis.LatticeVectors
    expansion: basis.Expansion
    wavelength: jax.Array
    theta: jax.Array
    phi: jax.Array
    in_plane_wavevector: jax.Array
    permittivities: list[jax.Array]
    thicknesses: list[jax.Array]
    layer_solve_results: list[fmm.LayerSolveResult]
    s_matrix: scattering.ScatteringMatrix

    @classmethod
    def from_model(
        cls,
        model: Model,
        config: Config
    ) -> "PreparedFMMaxModel":
        """
        Translate the MetaRCWA model into FMMax simulation objects
        """

        cfg = FMMaxConfig.from_config(config)

        # Resolve the wavelength-dependent material values and rasterise
        # the pattern onto the requested nx by ny grid
        model_spec = model.spec(nx=cfg.nx,
                                ny=cfg.ny)

        # MetaRCWA calls these vectors a1 and a2. FMMax calls 
        # the equivalent primitive lattice vectors u and v
        lattice_vectors = basis.LatticeVectors(
            u=torch_to_jax(model_spec.a1, dtype=cfg.real_dtype),
            v=torch_to_jax(model_spec.a2, dtype=cfg.real_dtype)
        )

        # MetaRCWA provides shape (Nw,), while kx0 and ky0 use
        # (Nw, Ntheta, Nphi). Add singleton dimension so that JAX 
        # reuses each wavelength across every angle
        # (Nw,) -> (Nw, 1,1)
        wavelength = torch_to_jax(model_spec.wavelength, dtype=cfg.real_dtype).reshape(-1,1,1)

        # MetaRCWA kx0 and ky0 have shape (Nw, Ntheta, Nphi) and are 
        # normalised by the free-space wavenumber.
        # FMMax expects physical in-plane wavevectors, so multiply by k0. 
        free_space_wavenumber = 2 * jnp.pi / wavelength

        kx = (
            torch_to_jax(model_spec.kx0, dtype=cfg.real_dtype)
            *
            free_space_wavenumber
        ) 

        ky = (
            torch_to_jax(model_spec.ky0, dtype = cfg.real_dtype)
            *
            free_space_wavenumber
        )

        # Stack the components on the final axis:
        # (Nw, Ntheta, Nphi) -> (Nw, Ntheta, Nphi, 2)
        # where the final dimension contains [kx,ky]
        # jnp.stack pairs each `kx` with its corresponding `ky`
        in_plane_wavevector = jnp.stack((kx,ky), axis=-1)

        incidence_eps = prepare_uniform_permittivity(
            model_spec.incidence.eps, complex_dtype=cfg.complex_dtype
        )

        transmission_eps = prepare_uniform_permittivity(
            model_spec.transmission.eps, complex_dtype=cfg.complex_dtype
        )

        layer_permittivities = []
        layer_thicknesses = []

        for layer in model_spec.layers:
            thickness = torch_to_jax(layer.thickness, dtype=cfg.real_dtype)
            # A scale thickness has shape ()
            layer_thicknesses.append(thickness)

            if isinstance(layer, HomogeneousLayer):
                # Output shape: (Nw,1,1,1,1)
                permittivity = prepare_uniform_permittivity(
                    layer.medium.eps,
                    complex_dtype=cfg.complex_dtype
                )
            elif isinstance(layer, PatternedLayer):
                # A patterned layer contains:
                # - a solid material
                # - a void/background material
                # - a spatial mask selecting between them
                # Output shape: (Nw,1,1,1,1)
                # Each material initially varies only with 
                # wavelength
                solid_eps = prepare_uniform_permittivity(
                    layer.medium_solid.eps,
                    complex_dtype=cfg.complex_dtype
                )
                void_eps = prepare_uniform_permittivity(
                    layer.medium_void.eps,
                    complex_dtype=cfg.complex_dtype
                )

                # Pattern shape: (Ngrid0, Ngrid1)
                pattern = torch_to_jax(
                    layer.pattern,
                    dtype = cfg.real_dtype
                )

                # Combine the two materials and pattern into one
                # spatial permittivity grid for the FMMax eigensolver
                # density = 1 selects the solid material
                # density = 0 selects the void material

                # Broadcasting combines:
                # solid_eps: (Nw,1,1,1,     1)
                # void_eps: (Nw,1,1,1,      1)
                # pattern:              (Ngrid0,Ngrid1)
                # result: (Nw,1,1,Ngrid0,Ngrid1)
                permittivity = utils.interpolate_permittivity(
                    permittivity_solid = solid_eps,
                    permittivity_void=void_eps,
                    density = pattern
                )

            else:
                # Raise error if the layers are not
                # supported by the layer classes translated above
                raise TypeError(
                    "Unsupported MetaRCWA layer type: "
                    f"{type(layer).__name__}"
                )
            layer_permittivities.append(permittivity)
            
        # Complete incidence to transmission stack
        permittivities = [
            incidence_eps,
            *layer_permittivities,
            transmission_eps
        ]
        # The two end media are semi-infinite and therefore don't have
        # finite propagation thickness.
        zero_thickness = jnp.asarray(0.0,
                                    dtype=cfg.real_dtype)
        
        thicknesses = [zero_thickness, *layer_thicknesses, zero_thickness]

        # Generate the reciprocal-space Fourier basis shared by every layer
        # Shape = (Nterms, 2)
        expansion = basis.generate_expansion(
            primitive_lattice_vectors = lattice_vectors,
            approximate_num_terms = cfg.approximate_num_terms,
            truncation = cfg.truncation
        )

        # Solve the electromagnetic modes independently in every medium 
        # finite layer
        layer_solve_results = [
            fmm.eigensolve_isotropic_media(
                wavelength=wavelength,
                in_plane_wavevector=in_plane_wavevector,
                primitive_lattice_vectors=lattice_vectors,
                permittivity=permittivity,
                expansion=expansion,
                formulation=cfg.formulation
            )
            for permittivity in permittivities
        ]

        # Combine the individual layer solutions into one scattering matrix
        # for the complete stack
        s_matrix = scattering.stack_s_matrix(
            layer_solve_results=layer_solve_results,
            layer_thicknesses = thicknesses
        )

        #cls(...) constructs and returns one PreparedFMMaxModel object
        return cls(
            config=cfg,
            model_spec=model_spec,
            lattice_vectors=lattice_vectors,
            expansion=expansion,
            wavelength=wavelength,
            in_plane_wavevector=in_plane_wavevector,
            incidence_eps=incidence_eps,
            transmission_eps=transmission_eps,
            permittivities=permittivities,
            thicknesses=thicknesses,
            layer_solve_results=layer_solve_results,
            s_matrix=s_matrix,
        )

def reflectance_and_transmittance_fmmax(
        s_matrix,
        layer_solve_results,
        polarization: str,
        thicknesses
):
    # Number of retained diffarction orders
    # FMMax stores two modal channels per diffraction order
    # index 0 has the first polarization block and index 1 the second
    n = layer_solve_results[0].expansion.num_terms

    # Zeroth-order plane wave excitation for s and p polarisation
    # First axis, axis 0: modal channel
    # which is diffraction order and polarisation block of output
    # Second axis, axis 1: independent incident source polarisation
    forward_amplitude_0 = jnp.zeros((2*n,2), dtype=complex)
    forward_amplitude_0 = forward_amplitude_0.at[0,0].set(1) #s
    forward_amplitude_0 = forward_amplitude_0.at[n,1].set(1) #p

    # Reflected zeroth order amplitude
    # s21 maps incident amplitudes to reflected amplitudes
    reflected_amplitude_0 = s_matrix.s21 @ forward_amplitude_0

    # Incident and reflected power flux by modal channel
    incident_flux, reflected_flux = fields.amplitude_poynting_flux(
        forward_amplitude=forward_amplitude_0,
        backward_amplitude=reflected_amplitude_0,
        layer_solve_result=layer_solve_results[0]
    )

    # s11 maps incident amplitudes to transmitted amplitudes
    transmitted_amplitude_0 = s_matrix.s11 @ forward_amplitude_0

    # No wave is incident from the transmission side
    transmitted_flux,_=fields.amplitude_poynting_flux(
        forward_amplitude=transmitted_amplitude_0,
        backward_amplitude=jnp.zeros_like(transmitted_amplitude_0),
        layer_solve_result=layer_solve_results[-1]
    )

    # Sum over every diffraction order and output polarisation
    incident_power = jnp.sum(incident_flux, axis=0)
    reflected_power = jnp.sum(reflected_flux, axis=0)
    transmitted_power = jnp.sum(transmitted_flux, axis=0)

    # Total reflectance for s incidence would be R[0]
    # Total reflectance for p incidence would be R[1]
    R = -reflected_power / incident_power 
    T = transmitted_power / incident_power

    return{
        "R": R,
        "T": T
    }