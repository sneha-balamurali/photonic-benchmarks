import jax
import torch
from typing import Any
from dataclasses import dataclass

# JAX disables float64 by default. Enabling it 
# allows the adapter to run with float64 precision
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from fmmax import (basis,
                   sources,
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
    incidence_eps: jax.Array
    transmission_eps:jax.Array
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

        # theta and phi
        theta = torch_to_jax(model.source.theta,dtype=cfg.real_dtype).reshape(-1)

        phi = torch_to_jax(model.source.phi,dtype=cfg.real_dtype).reshape(-1)

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

                # MetaRCWA/MetaShapes mask: (..., Ny, Nx)
                # FMMax spatial convention: (..., Nx, Ny)
                # Swap only the two trailing spatial axes.
                pattern = torch_to_jax(
                    layer.pattern,
                    dtype = cfg.real_dtype
                )

                pattern = jnp.swapaxes(pattern,-2,-1)

                # Broadcasting combines:
                # solid_eps: (Nw,1,1,1,1)
                # void_eps: (Nw,1,1,1,1)
                # pattern:          (Ny,Nx)
                # result: (Nw,1,1,Nx,Ny)

                # Construct the patterned permittivity using MetaRCWA's material-mixing
                # convention so both solvers receive the same permittivity grid:
                #
                #   pattern = 1 -> solid permittivity
                #   pattern = 0 -> void permittivity
                #   0 < pattern < 1 -> direct linear interpolation of permittivity
                #
                # FMMax's utils.interpolate_permittivity() instead interpolates the
                # complex refractive index. We intentionally use direct permittivity
                # interpolation here to avoid introducing different material preprocessing
                # into the solver comparison.

                permittivity = (
                    pattern * solid_eps
                    + (1.0 - pattern) * void_eps
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
            theta=theta,
            phi=phi,
            in_plane_wavevector=in_plane_wavevector,
            incidence_eps=incidence_eps,
            transmission_eps=transmission_eps,
            permittivities=permittivities,
            thicknesses=thicknesses,
            layer_solve_results=layer_solve_results,
            s_matrix=s_matrix,
        )

def reflectance_and_transmittance_fmmax(
        model: PreparedFMMaxModel
): 
    """
    1. Construct the incident modal amplitude with column 0
    representing the s input and column 1 the p input
    2. Apply the scattering matrix that relates incoming forward 
    and backward modal amplitudes to the outgoing amplitudes on the 
    incidence and transmission sides.
    3. Convert the modal ampltidues to the Poynting flux, sum over the 
    output channels and divide the output by incident power to get 
    the reflectance and transmittance. 

    The current result extraction supports the benchmark's
    normal-incidence, zero-azimuth source.
    """
    # Find number of retained Fourier/diffarction orders
    # FMMax stores two transverse modal channels per diffraction order
    # Channels 0:n contain the first transverse modal block  
    # Channels n:2n contain the second transverse modal block
    no_fourier = model.layer_solve_results[0].expansion.num_terms

    # wavelength, theta and phi from sources:
    wavelength = model.wavelength
    theta = model.theta
    phi = model.phi

    # Parameter sweep shape
    batch_shape = (wavelength.shape,
                   theta.shape,
                   phi.shape)    

    # Incidence permittivity
    n_inc = jnp.sqrt(model.incidence_eps)

    # Construct incident Ex and Ey for s and p polarisation
    # When z component not defined, means it is 0.

    # s polarisation:
    Ex_s = jnp.broadcast_to(-jnp.sin(phi), 
                            batch_shape)
    Ey_s = jnp.broadcast_to(jnp.cos(phi),
                            batch_shape)

    Hx_s = jnp.broadcast_to(-n_inc * jnp.cos(theta)*jnp.cos(phi),
                            batch_shape)
    Hy_s = jnp.broadcast_to(-n_inc * jnp.cos(theta)*jnp.sin(phi),
                            batch_shape)
    Hz_s = jnp.broadcast_to(n_inc * jnp.sin(theta),
                            batch_shape)

    # p polarisation:
    Ex_p = jnp.broadcast_to(jnp.cos(theta) * jnp.cos(phi),
                            batch_shape)
    Ey_p = jnp.broadcast_to(jnp.cos(theta) * jnp.sin(phi),
                            batch_shape)
    Ez_p = jnp.broadcast_to(-jnp.sin(theta),
                            batch_shape)

    Hx_p= jnp.broadcast_to(-n_inc * jnp.sin(phi),
                           batch_shape)
    Hy_p = jnp.broadcast_to(n_inc * jnp.cos(phi),
                            batch_shape)

    # Combine the two incident polarisations into one axis
    # New shape: (Nw, Ntheta, Nphi, 2)
    # [0]: s polarisation; [1]: p polarisation
    Ex = jnp.stack((Ex_s,Ex_p),axis=-1)
    Ey = jnp.stack((Ey_s, Ey_p), axis=-1)

    Hx = jnp.stack((Hx_s, Hy_s), axis=-1)
    Hy = jnp.stack((Hy_s, Hy_p), axis=-1)

    # Add two spatial axes before `num_fields`
    # FMMax's `amplitudes_for_fields` requires
    # shape (...,nx,ny,num_fields)
    

    # Create two incidence excitations
    # Shape (2*n modal channels, 2 incidence sources)
    # Initially everthing 0 - no mode excited

    forward_amplitude_0 = jnp.zeros((2*n,2), dtype=complex)

    # For arbitrary azimuthal, the fixed channels are not
    # necessarily the physical s/p directions. 

    # Under normal-incidence convention, column 0 is
    # s polarised incident case, column 1 is p polarised incidence
    # case. Channel 0 and n are zeroth order plane wave excitation.
    forward_amplitude_0 = forward_amplitude_0.at[0,0].set(1) #s
    forward_amplitude_0 = forward_amplitude_0.at[n,1].set(1) #p

    # Reflected zeroth order amplitude
    # s21 maps incident amplitudes to reflected amplitudes
    reflected_amplitude_0 = s_matrix.s21 @ forward_amplitude_0

    # Incident and reflected power flux by modal channel
    incident_flux, reflected_flux = fields.amplitude_poynting_flux(
        forward_amplitude=forward_amplitude_0,
        backward_amplitude=reflected_amplitude_0,
        # Incident and reflected wave in the incidence medium so 
        # need the results of the layer eigensolve in [0]
        # the incidence medium
        layer_solve_result=layer_solve_results[0]
    )

    # s11 maps incident amplitudes to transmitted amplitudes
    transmitted_amplitude_n = s_matrix.s11 @ forward_amplitude_0


    transmitted_flux,_=fields.amplitude_poynting_flux(
        forward_amplitude=transmitted_amplitude_n,
        # No wave is incident from the transmission side
        backward_amplitude=jnp.zeros_like(transmitted_amplitude_n),
        # Transmitted waves in the transmission medium so 
        # need the results of the layer eigensolve in [-1]
        # the transmission medium
        layer_solve_result=layer_solve_results[-1]
    )

    # Sum over every diffraction order and output polarisation
    # shape is (Nw, Ntheta, Nphi, 2*num_terms, 2)
    # axis -2 represents a fourier order and one of the 2 independent
    # polarisation/mode combination
    incident_power = jnp.sum(incident_flux, axis=-2)
    reflected_power = jnp.sum(reflected_flux, axis=-2)
    transmitted_power = jnp.sum(transmitted_flux, axis=-2)

    # Total reflectance for s incidence would be R[0]
    # Total reflectance for p incidence would be R[1]
    R = -reflected_power / incident_power 
    T = transmitted_power / incident_power

    return{
        "R": R,
        "T": T
    }

def run_fmmax(
    model: Model,
    config: Config,
) -> dict[str, jax.Array]:
    """Prepare the model, run FMMax and return total powers:
    - Rs -> total reflected power for s incidence
    - Rp -> total reflected power for p incidnece
    - Ts -> total transmitted power for s incidence
    - Tp -> total transmitted power for p incidence
    """

    prepared = PreparedFMMaxModel.from_model(
        model=model,
        config=config,
    )

    result = reflectance_and_transmittance_fmmax(
        s_matrix=prepared.s_matrix,
        layer_solve_results=(
            prepared.layer_solve_results
        ),
    )

    R = result["R"]
    T = result["T"]

    return {
        "Rs": R[..., 0],
        "Rp": R[..., 1],
        "Ts": T[..., 0],
        "Tp": T[..., 1],
        "requested_orders": (
            prepared.config.approximate_num_terms
        ),
        "actual_orders": (
            prepared.expansion.num_terms
        ),
        "basis_coefficients": (
            prepared.expansion.basis_coefficients
        ),
    }