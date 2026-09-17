from dataclasses import dataclass
from typing import Any
import math

import S4
import torch
from metarcwa import Model

from src.config import Config
from .config import S4Config
from .geometry import add_shape, metashape_from_layer

from metarcwa.model.layer import HomogeneousLayer, PatternedLayer

def tensor_to_tuple(value:torch.Tensor) -> tuple[float,float]:
    """Convert a two-component PyTorch tensor into an ordinary
    Python tuple.

    MetaRCWA stores each lattice vector as a PyTorch tensor 
    with shape (2,) but S4's python interface expects 
    something like ((a1_x,a1_y), (a2_x,a2_y))

    Gradient tracking and GPU storage are not meaningful to S4, 
    so the tensor is detached and moved to CPU before extracting
    its values.
    """

    # Check that S4 lattice vectors only contains 2 componenets
    if value.numel() != 2:
        raise ValueError(
            "A S4 lattice vector must contain exactly 2 components."
            f"Recieved shape {tuple(value.shape)}."
        )

    flattened = value.detach().cpu().reshape(-1)

    return (float(flattened[0]),float(flattened[1]))

def tensor_to_float(value: torch.Tensor) -> float:
    """Convert a one-value PyTorch tensor into a Python float."""

    if value.numel() != 1:
        raise ValueError(
            "Expected a tensor containing exactly one value. "
            f"Received shape {tuple(value.shape)}."
        )

    return float(value.detach().cpu().item())

def permittivity_at_wavelength(
    value: torch.Tensor,
    wavelength_index: int
) -> complex:
    """Select one wavelength's permittivity as a Python complex number.

    MetaRCWA evaluates an isotropic material at every requested wavelength,
    producing a tensor with shape (Nw,). S4 has one active material value at 
    a time, so we select the value for one wavelength.

    Parameters:
    -----------
    value:
        Complex MetaRCWA permittivity tensor with shape (Nw)
    wavelength_index:
        Position of the wavelength to select. Index zero selects the first
        wavelength.
    """

    flattened = value.detach().cpu().reshape(-1)

    return complex(flattened[wavelength_index].item())

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
    wavelength_index:
        Position of the selected wavelength in the MetaRCWA wavelength sweep.
        Index zero selects the first wavelength.
    wavelength:
        Selected wavelength as an Python float. Uses the same length unit as
        the model geometry.
    """

    config: S4Config
    model: Model
    model_spec: Any
    lattice_vectors: tuple[
        tuple[float,float],
        tuple[float,float]
    ]
    wavelength_index: int
    wavelength: float
    simulation: Any

    @classmethod
    def from_model(
        cls, 
        model: Model,
        config: Config,
        wavelength_index: int=0
        ) -> "PreparedS4Model":
        """Translate a MetaRCWA model into S4 simulation objects.

        This method
        
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

        # MetaRCWA stores all the requested wavelengths in a PyTorch tensor
        # e.g. tensor([500,600])
        # One S4 simulation object uses one frequency at a time, so select
        # the wavelength indicated by wavelength_index.

        wavelengths = (model_spec.wavelength
                       .detach()
                       .cpu()
                       .reshape(-1)
                       )

        number_of_wavelengths = wavelengths.numel()
        
        # Check the index and give a clear explanation if 
        # there is an indexing error
        if not 0<= wavelength_index < number_of_wavelengths:
            raise IndexError(
                "wavelength_index is outside the model's wavelength array. "
                f"Received {wavelength_index}, but model contains "
                f"{number_of_wavelengths} wavelength values."
            )

        # item() extracts one number from a one value PyTorch tensor
        # float() converts that number into a float
        # which is what the S4 Python interface expects.
        wavelength = float(
            wavelengths[wavelength_index].item()
        )

        # MetaRCWA stores lattice vectors as PyTorch tensors
        # S4 expects something of the form:
        # ((a1_x,a1_y), (a2_x,a2_y))
        # A nested tuple with two 2D lattice vectors
        lattice_vectors = (
            tensor_to_tuple(model_spec.a1),
            tensor_to_tuple(model_spec.a2)
        )

        simulation = S4.New(
            Lattice=lattice_vectors,
            NumBasis=cfg.requested_num_basis
        )

        # S4 expects frequency = 1 / wavelength, rather than 
        # wavelength itself or angular frequency.
        # All lengths must use one consistent unit. In this model,
        # both geometry and wavelength use nanometres, so the
        # frequency is expressed in inverse nanometres.
        simulation.SetFrequency(
            1.0 / wavelength
        )

        simulation.SetOptions(LatticeTruncation= 
                            cfg.lattice_truncation)

        # S4 represents the semi-infinite incidence medium as the first
        # layer with zero thickness
        simulation.SetMaterial(
            Name="incidence",
            Epsilon=permittivity_at_wavelength(
                model_spec.incidence.eps,
                wavelength_index
            )
        )

        # Layer order follows the direction of incidence,
        # from top to bottom.
        simulation.AddLayer(
            Name="incidence",
            Thickness=0.0,
            Material="incidence"
        )

        # Add the finite layers in incidence to transmission order
        # MetaRCWA model_spec defines layer 0 from the first finite layer
        # The semi-infinite incidence and transmission layers are defined
        # seperately 
        for layer_index, layer in enumerate(model_spec.layers):
            layer_name = f"layer_{layer_index}"

            if isinstance(layer, HomogeneousLayer):
                # A homogeneous layer contains only one material 
                # and doesn't require an analytical geometry region
                material_name = f"layer_{layer_index}_material"

                simulation.SetMaterial(
                    Name = material_name,
                    Epsilon=permittivity_at_wavelength(
                        layer.medium.eps,
                        wavelength_index
                    )
                )

                simulation.AddLayer(
                    Name=layer_name,
                    Thickness=tensor_to_float(layer.thickness),
                    Material=material_name
                )
            elif isinstance(layer, PatternedLayer):
                # S4 creates a patterned layer by first filling the
                # entire layer with its background material. Geometry
                # regions made from different material will be inserted 
                # in next implementation

                solid_material_name = f"layer_{layer_index}_solid"
                void_material_name = f"layer_{layer_index}_void"

                simulation.SetMaterial(
                    Name=solid_material_name,
                    Epsilon = permittivity_at_wavelength(
                        layer.medium_solid.eps,
                        wavelength_index
                    )
                )

                simulation.SetMaterial(
                    Name=void_material_name,
                    Epsilon = permittivity_at_wavelength(
                        layer.medium_void.eps,
                        wavelength_index
                    )
                )

                # Begin by filling the layer with the 
                # void/background material
                simulation.AddLayer(
                    Name=layer_name,
                    Thickness=tensor_to_float(layer.thickness),
                    Material=void_material_name
                )

                # The resolved layer contains a raster mask, 
                # while the unresolved layer contains the original
                # MetaShapes geometry.
                original_layer = model.stack.layers[layer_index]

                shape = metashape_from_layer(original_layer)

                # Insert the solid analytical geometry into the background
                # layer. The mapping geometry.py selects add_rectangle()
                # for the current Rectangle. 
                add_shape(
                    simulation,
                    layer_name=layer_name,
                    material_name=solid_material_name,
                    shape=shape,
                    lattice_vectors=lattice_vectors
                )

            else:
                raise TypeError(
                    "Unsupported MetaRCWA layer type:"
                    f"{type(layer).__name__}"
                )

        # Set the final semi-infinite transmission medium as the final
        # zero thickness layer at the end of the stack
        simulation.SetMaterial(
            Name="transmission",
            Epsilon=permittivity_at_wavelength(
                model_spec.transmission.eps,
                wavelength_index
            )
        )

        simulation.AddLayer(
            Name="transmission",
            Thickness=0.0,
            Material="transmission"
        )

        return cls(
            config=cfg,
            model_spec=model_spec,
            lattice_vectors=lattice_vectors,
            simulation=simulation,
            wavelength_index=wavelength_index,
            model=model,
            wavelength=wavelength
        )

def solve_s4_total_power(
    prepared: PreparedS4Model,
    theta_rad: float,
    phi_rad: float,
    polarization: str,
) -> tuple[float,float]:
    """Solve one S4 excitation and return total reflection and transmission.

    This is the first, deliberately small S4 solving function. It handles
    exactly one:

    - selected wavelength, already stored in `prepared`
    - polar incidence angle
    - azimuthal angle
    - incidence polarization

    The returned powers are summed over all the retained diffraction orders.
    Pre-order power extraction will be added after this total power route
    has been added. 

    Parameters
    ----------
    prepared:
        S4 simulation constructed by PreparedS4Model.from_model().
        It already contains the lattice, materials, finite layers,
        analytical geometry and selected wavelength.
    theta_rad:
        Polar incidence angle in radians, measured away from the surface 
        normal. Zero means normal incidence. 
    phi_rad: 
        Azimuthal angle in radians, measured in the x-y plane. 
    polarizatuion:
        Polarization of the incident plane wave. Must be with s or p.
    
    Returns
    -------
    reflection:
        Total reflected power divided by incident power.
    transmission:
        Total transmitted power divided by incident power.

    Notes
    -----
    Reflection uses a minus sign because reflected power travels in the
    negative z direction. S4 therefore reports its backward flux with a 
    negative real part.
    """

    # Reject spelling mistakes or other conventions like te or tm
    if polarization not in {'s', 'p'}:
        raise ValueError(
            "polarization must be either 's' or 'p'. "
            f"Recieved {polarization!r}."
        )

    # MetaRCWA describes theta and phi in radians while S4's
    # SetExcitationPlaneWave() expects the physical polar and 
    # azimuthal angles in degrees.
    theta_degrees = math.degrees(theta_rad)
    phi_degrees = math.degrees(phi_rad)

    # Run one pure polarization experiment:
    # s experiment: s amplitude = 1 and p amplitude = 0
    # p experiment: s amplitude = 0 and p amplitude = 1
    if polarization == 's':
        s_amplitude = 1.0 + 0.0j
        p_amplitude = 0.0 + 0.0j
    else:
        s_amplitude = 0.0 + 0.0j
        p_amplitude = 1.0 + 0.0j
    
    # S4 identifies an incident diffraction order by its position in the
    # basis returned by GetBasisSet(). 
    orders = tuple(prepared.simulation.GetBasisSet())

    # Get the position corresponding to (m,n) = (0,0)
    try: 
        incident_order_index = orders.index((0,0))
    except ValueError as error:
        raise RuntimeError(
            "S4 did not retain the incident diffraction order (0,0)."
        ) from error

    # Illuminate the first layer of the S4 stack with one plane wave
    # The first angle is the polar angle from the normal
    # The second angle is the azimuthal rotation
    prepared.simulation.SetExcitationPlanewave(
        IncidenceAngles=(
            theta_degrees,
            phi_degrees
        ),
        sAmplitude = s_amplitude,
        pAmplitude = p_amplitude,
        Order = incident_order_index
    )

    # In the incidence layer:
    # forward flux = power travelling into the structure,
    # more specifically the forward component of the complex
    # Poynting vector.
    # backward flux = reflected power travelling back out
    incident_flux, reflected_flux = (
        prepared.simulation.GetPowerFlux(
            Layer="incidence",
            zOffset=0.0
        )
    )

    # In the final transmission layer:
    # The forward flux travels out through the bottom of the structure.
    # No source is entering from this side, so its backward flux is not
    # required here. 
    transmitted_flux, _ = prepared.simulation.GetPowerFlux(
        Layer="transmission",
        zOffset=0.0
    )

    # GetPowerFlux() returns complex forward and backward components of 
    # the z directed Poynting flux. Reflection and transmission are
    # calculated from their real parts.
    incident_power = incident_flux.real
    reflected_power = reflected_flux.real
    transmitted_power = transmitted_flux.real

    if incident_power <=0:
        raise RuntimeError(
            "S4 returned non-positive incident power, so reflection and "
            f"transmission cannot be normalised. Received {incident_power}."
        )
    # Reflected flux is negative because it propagates in the negative
    # z direction. The minus sign converts it into a positive reflectance. 
    reflection = -reflected_power / incident_power

    # Transmitted flux travels in the positive z direction and so doesn't
    # require a sign change. 
    transmission = transmitted_power / incident_power

    return reflection, transmission

def run_s4(
    model: Model,
    config: Config,
) -> dict[str, Any]:
    """Return total reflected and transmitted power."""

    wavelengths = model.source.wavelength.detach().cpu().reshape(-1)
    theta_values = model.source.theta.detach().cpu().reshape(-1)
    phi_values = model.source.phi.detach().cpu().reshape(-1)

    result_shape = (
        wavelengths.numel(),
        theta_values.numel(),
        phi_values.numel(),
    )

    Rs = torch.empty(result_shape, dtype=torch.float64)
    Rp = torch.empty(result_shape, dtype=torch.float64)
    Ts = torch.empty(result_shape, dtype=torch.float64)
    Tp = torch.empty(result_shape, dtype=torch.float64)

    for wavelength_index in range(wavelengths.numel()):
        prepared = PreparedS4Model.from_model(
            model=model,
            config=config,
            wavelength_index=wavelength_index,
        )

        for theta_index, theta in enumerate(theta_values):
            for phi_index, phi in enumerate(phi_values):
                index = (
                    wavelength_index,
                    theta_index,
                    phi_index,
                )

                Rs[index], Ts[index] = solve_s4_total_power(
                    prepared=prepared,
                    theta_rad=float(theta.item()),
                    phi_rad=float(phi.item()),
                    polarization="s",
                )

                Rp[index], Tp[index] = solve_s4_total_power(
                    prepared=prepared,
                    theta_rad=float(theta.item()),
                    phi_rad=float(phi.item()),
                    polarization="p",
                )

    cfg = S4Config.from_config(config)
    orders = tuple(prepared.simulation.GetBasisSet())

    return {
        "Rs": Rs,
        "Rp": Rp,
        "Ts": Ts,
        "Tp": Tp,
        "requested_orders": cfg.requested_num_basis,
        "actual_orders": len(orders),
        "basis_coefficients": torch.tensor(
            orders,
            dtype=torch.int64,
        ),
    }

    