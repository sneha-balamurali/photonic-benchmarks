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

    return converted.reshape(-1,1,1,1,1)

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
        theta = torch_to_jax(model.source.theta,dtype=cfg.real_dtype).reshape(1,-1,1)

        phi = torch_to_jax(model.source.phi,dtype=cfg.real_dtype).reshape(1,1,-1)

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
                # pattern:          (Nx,Ny)
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

def construct_incidence_amplitudes(
        prepared: PreparedFMMaxModel
): 
    """
    Construct s- and p-polarized plane-wave amplitudes
    """

    # wavelength, theta and phi from sources:
    wavelength = prepared.wavelength   #(Nw,1,1)
    theta = prepared.theta             #(1,Ntheta,1)
    phi = prepared.phi                 #(1,1,Nphi)

    # Parameter sweep shape
    # (Nw, Ntheta, Nphi)
    batch_shape = (wavelength.shape[0],
                   theta.shape[1],
                   phi.shape[2])    

    # Incidence permittivity
    # Shape of model.incidence_eps is (Nw,1,1,1,1)
    # Need to remove final 1x1 material-grid axes before
    # broadcasting across wavelength, theta, phi.
    # New shape: (Nw,1,1)
    n_inc = jnp.sqrt(
        jnp.squeeze(
            prepared.incidence_eps,
            axis=(-2,-1)
        )
    )

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

    Hx = jnp.stack((Hx_s, Hx_p), axis=-1)
    Hy = jnp.stack((Hy_s, Hy_p), axis=-1)

    # FMMax's `amplitudes_for_fields` requires
    # shape (...,nx,ny,num_fields)
    # This is used to sample the incidence field for
    # `amplitudes_for_fields()`

    # Use the configured real-space resolution to sample 
    # the incident field
    field_nx = prepared.config.nx
    field_ny = prepared.config.ny

    # Generate the x,y position of every point where the incident
    # field will be sampled and used to evaluate the spatial phase 
    # of the fields below.
    # x,y.shape: (field_nx,field_ny)
    x,y = basis.unit_cell_coordinates(
        primitive_lattice_vectors = prepared.lattice_vectors,
        # Creates the fractional spacing to sample,
        # determines sampling density
        shape = (field_nx,field_ny),
        # Number of unit cells to sample along a1 and a2
        num_unit_cells = (1,1)
    )

    # Want to calculate kx*x + ky * y for every wavelength,
    # theta, phi and x,y grid point so need to reshape kx and ky
    # Add two singelton spatial axes at the end
    # Shape: (Nw,Ntheta,Nphi,1,1)
    kx = prepared.in_plane_wavevector[...,0, None, None]
    ky = prepared.in_plane_wavevector[...,1, None, None]

    phase = jnp.exp(1j * (kx * x + ky * y))

    # Calculate complete electric and magnetic field 
    # Multiple the periodic envelope calculated above
    # with the Bloch phase

    # Ex.shape: (Nw,Ntheta,Nphi,2) -> 
    # (Nw,Ntheta,Nphi,field_x,field_y,2)
    # phase: (Nw, Ntheta,Nphi,field_x,field_y) -> 
    # (Nw, Ntheta,Nphi,field_x,field_y,1)
    # Result: (Nw,Ntheta,Nphi,field_x,field_y,2)
    Ex = Ex[...,None,None,:]*phase[...,None]
    Ey = Ey[...,None,None,:]*phase[...,None]
    Hx = Hx[...,None,None,:]*phase[...,None]
    Hy = Hy[...,None,None,:]*phase[...,None]

    incidence_eigensolve = prepared.layer_solve_results[0]

    forward_amplitudes,backward_amplitudes = (
        sources.amplitudes_for_fields(
            ex = Ex,
            ey = Ey,
            hx = Hx,
            hy = Hy,
            layer_solve_result = incidence_eigensolve,
            brillouin_grid_axes = None
        )
    )

    return forward_amplitudes, backward_amplitudes

def fmmax_reflectance_and_transmittance(
    prepared: PreparedFMMaxModel,
) -> dict[str, Any]:
    """Prepare the model, run FMMax and return total powers:
    - Rs -> total reflected power for s incidence
    - Rp -> total reflected power for p incidnece
    - Ts -> total transmitted power for s incidence
    - Tp -> total transmitted power for p incidence
    """
    s_matrix = prepared.s_matrix
    forward_amplitudes, source_backward_amplitudes = (
        construct_incidence_amplitudes(
            prepared = prepared
        )
    )

    # Incident and reflected wave in the incidence medium so need
    # the results of the layer eigensolve in [0] i.e. the incidence
    # medium
    incidence_layer_solve_result = prepared.layer_solve_results[0]
    reflected_amplitudes = s_matrix.s21 @ forward_amplitudes

    # Incidenct and reflected power flux by modal channel
    incident_flux, reflected_flux = (
        fields.amplitude_poynting_flux(
            forward_amplitude = forward_amplitudes,
            backward_amplitude = reflected_amplitudes,
            layer_solve_result = incidence_layer_solve_result
        )
    )
    
    # s11 maps incident amplitudes to transmitted amplitudes
    transmitted_amplitudes = s_matrix.s11 @ forward_amplitudes
    
    
    transmitted_flux,_=fields.amplitude_poynting_flux(
            forward_amplitude=transmitted_amplitudes,
            # No wave is incident from the transmission side
            backward_amplitude=jnp.zeros_like(transmitted_amplitudes),
            # Transmitted waves in the transmission medium so 
            # need the results of the layer eigensolve in [-1]
            # the transmission medium
            layer_solve_result=prepared.layer_solve_results[-1]
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
        "source_backward_residual": jnp.max(
            jnp.abs(source_backward_amplitudes)
),
    }

def run_fmmax(
    model: Model,
    config: Config,
) -> dict[str, Any]:
    """Translate the shared inputs, run FMMax and return the results."""

    prepared = PreparedFMMaxModel.from_model(
        model=model,
        config=config,
    )

    return fmmax_reflectance_and_transmittance(
        prepared=prepared
    )