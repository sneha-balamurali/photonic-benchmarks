from dataclasses import dataclass
from typing import Any
import math

import S4
import torch
from metarcwa import Model

from src.config import Config
from src.s4.config_prev import S4Config
from s4.geometry_prev import add_shape, metashape_from_layer

from metarcwa.model.layer import HomogeneousLayer, PatternedLayer

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

def solve_s4_power_by_order(
    prepared: PreparedS4Model,
    theta_rad: float,
    phi_rad: float,
    polarization: str,
) -> tuple[
    tuple[tuple[int,int], ...],
    torch.Tensor,
    torch.Tensor
]:
    """Return reflected and transmitted power for every S4 order.

    This function handles one selected wavelength, one incidence angle 
    and one incident polarization. Unlike `solve_s4_total_power()`, it keeps
    each diffraction order seperate.

    Parameters
    ----------
    prepared:
        S4 simulation created by `PreparedS4Model.from_model()`
    theta_rad:
        Polar incidence angle in radians, measured from the normal.
    phi_rad:
        Azimuthal incidence angle in radians.
    polarization:
        Incident polarization, either `s` or `p`.

    Returns
    -------
    orders:
        S4 diffraction-order pairs in their retained order.
        e.g ((0,0), (0,-1), (-1,0), ...)
    reflection_by_order:
        Reflected power in each order, divided by incident power. 
        Shape: (Norders,)
    transmission_by_order:
        Transmitted power in each order, divided by incident power
        Shape: (Norders,)

    Notes
    -----
    The same integer position refers to the same diffraction order in 
    all three returned values. For example, if orders[3] == (1,0), then 
    reflection_by_orders[3] is reflected power in order (1,0).
    """

    if polarization not in {"s", "p"}:
        raise ValueError(
            "polarization must be either 's' or 'p'. "
            f"Received {polarization!r}."
        )

    theta_degrees = math.degrees(theta_rad)
    phi_degrees = math.degrees(phi_rad)

    # Select one pure incident polarization
    if polarization == "s":
        s_amplitude = 1.0 + 0.0j
        p_amplitude = 0.0 + 0.0j
    else:
        s_amplitude = 0.0 + 0.0j
        p_amplitude = 1.0 + 0.0j

    #GetBasisSet() lists the retained physical diffraction order pairs
    #GetPowerFluxByOrders() returns its results in this same order
    orders = tuple(
        prepared.simulation.GetBasisSet()
    )

    try:
        incident_order_index = orders.index((0,0))
    except ValueError as error:
        raise RuntimeError(
            "S4 did not retain the incident diffraction order (0,0)"
        ) from error

    prepared.simulation.SetExcitationPlanewave(
        IncidenceAngles=(
            theta_degrees,
            phi_degrees
        ),
        sAmplitude=s_amplitude,
        pAmplitude=p_amplitude,
        Order=incident_order_index
    )

    # Use the total forward flux in the incidence medium as the
    # normalisation value for every outgoing diffraction order.
    incident_flux, _ = prepared.simulation.GetPowerFlux(
        Layer = "incidence",
        zOffset=0.0
    )

    incident_power = incident_flux.real

    if incident_power <=0:
        raise RuntimeError(
            "S4 returned non-positive incident power, so diffraction "
            f"powers cannot be normalised. Received {incident_power}."
        )

    # Each item returnd by GetPowerByFluxOrder() has the form:
    # (forward_flux, backward_flux)
    # There is one item for every pair in orders.
    incidence_flux_by_order = (
        prepared.simulation.GetPowerFluxByOrder(
            Layer="incidence",
            zOffset=0.0
        )
    )

    transmission_flux_by_order = (
        prepared.simulation.GetPowerFluxByOrder(
            Layer="transmission",
            zOffset=0.0
        )
    )

    # In the incidence medium, the outgoing reflected wave is the 
    # backward component. You take the negative because its signed z-directed
    # Poynting flux is negative. 
    reflection_by_order = torch.tensor(
        [ -backward_flux.real / incident_power
        for _, backward_flux in incidence_flux_by_order],
        dtype=torch.float64
    )

    # In the transmission medium, the outgoing transmitted wave is the
    # forward component so you don't need to change the sign
    transmission_by_order = torch.tensor(
        [ forward_flux.real / incident_power
        for forward_flux, _ in transmission_flux_by_order],
        dtype = torch.float64
    )

    return(orders, reflection_by_order, transmission_by_order)

@dataclass
class S4DiffractionResult:
    """Diffraction order powers returned by S4
    
    S4 is run twice at every wavelength and angle:
    - once with an s-polarised incident wave
    - once with a p-polarised incident wave
    
    Each output tensor stores the power in every retained
    diffraction order. The last tensor axis corresponds to
    these orders.

    Attributes
    ----------
    orders:
        Retained S4 diffraction-order pairs (m,n). If 
        orders[3] == (1,-1), then index 3 on every power tensor
        contains the power in diffraction order (1,-1).
    reflection_s_incident:
        Reflected power per order for an s-polarised incident 
        wave. Shape (Nw, Ntheta, Nphi, Norders)
    reflection_p_incident:
        Reflected power per order for a p-polarised incident 
        wave. Shape (Nw, Ntheta, Nphi, Norders)
    transmission_s_incident:
        Transmitted power per order for a s-polarised incident
        wave. Shape (Nw, Ntheta, Nphi, Norders)
    transmission_p_incident:
        Transmitted power per order for a p-polarised incident
        wave. Shape (Nw, Ntheta, Nphi, Norders)

    Notes
    -----
    S4's GetPowerFluxByOrder() reports total power in each diffraction
    order. It doesn't seperate the outgoing order into s and p 
    polarised components.
    """

    orders: tuple[tuple[int,int],...]

    reflection_s_incident: torch.Tensor
    reflection_p_incident: torch.Tensor
    transmission_s_incident: torch.Tensor
    transmission_p_incident: torch.Tensor

    def powers_for_order(self,
                         order: tuple[int,int]
                    ) -> tuple[torch.Tensor,
                               torch.Tensor,
                               torch.Tensor,
                               torch.Tensor]:
        """Return s and p incident powers for one diffraction order.

        Paramters
        ---------
        order: 
            Diffraction-order pair (m,n) such as (0,0),
            (1,0), (1,-1)
        
        Returns
        -------
        Rs, Rp, Ts, Tp:
            Power tensors with shape (Nw, Ntheta, Nphi).
            Where s and p in the subscript indicate the polarisation
            of the incident wave. S4's per order flux doesn't seperate
            the output wave into co and cross polarised parts. 
        """

        # orders is a tuple such as:
        # ((0,0), (0,-1), (-1,0),...)
        # Find the integer position of the requested pair
        try:
            order_index = self.orders.index(order)
        except ValueError as error:
            raise ValueError(
                f"Diffraction order {order!r} was not retained by S4. "
                f"Retained orders: {self.orders!r}"
            ) from error

        # Each complete power tensor has the shape:
        # (Nw, Ntheta, Nphi, Norders)
        # order_index selects one diffraction order from the final axis
        Rs=self.reflection_s_incident[...,order_index]
        Rp=self.reflection_p_incident[...,order_index]
        Ts=self.transmission_s_incident[...,order_index]
        Tp=self.transmission_p_incident[...,order_index]

        return Rs, Rp, Ts, Tp

def run_s4_diffraction(
    model: Model,
    config: Config,
) -> S4DiffractionResult:
    """Run S4 over every wavelength, angle and incident polarization.

    MetaRCWA stores wavelength, theta, phi as independent sweep axes. 
    S4 solves one wavelength and one angle pair at a time, so this
    function loops over every combination explicitly. 

    At each wavelength and angle, S4 is run twice:

    - once with an s polarised incident wave
    - once with a p polarised incident wave

    The power is retained seperately for every S4 diffraction order.

    Parameters
    ----------
    model:
        Solver independent MetaRCWA model containing the physical stack,
        wavelength values and incidence angles.
    config:
        Solver independent numerical configuration.
    
    Returns
    -------
    S4DiffractionResult:
        Reflected and transmitted power for every wavelength, angle,
        incident polarisation and retained diffraction order.

        Each power tensor has the shape:
        (Nw, Ntheta, Nphi, Norders)
    """

    # MetaRCWA stores wavelength, theta, phi as PyTorch tensors
    # reshape(-1) is used to ensure that each one is treated as a 
    # 1D sweep
    wavelengths = (
        model.source.wavelength
        .detach()
        .cpu()
        .reshape(-1)
    )

    theta_values = (
        model.source.theta
        .detach()
        .cpu()
        .reshape(-1)
    )

    phi_values = (
        model.source.phi
        .detach()
        .cpu()
        .reshape(-1)
    )

    # Prepare the first wavelength so we can ask S4 which diffraction 
    # order it has retained. The number and order of these pairs determine
    # the final axis of every result tensor.

    first_prepared = PreparedS4Model.from_model(
        model=model,
        config=config,
        wavelength_index=0
    )

    orders = tuple(first_prepared.simulation.GetBasisSet())

    number_of_orders = len(orders)
    number_of_wavelengths = wavelengths.numel()
    number_of_thetas = theta_values.numel()
    number_of_phis = phi_values.numel()

    result_shape = (
        number_of_wavelengths,
        number_of_thetas,
        number_of_phis,
        number_of_orders
    )

    # Create empty tensors that will be filled when looped through
    # each simulation. The s and p suffix describes the incident 
    # polarisation. S4's GetPowerFluxByOrder() doesn't seperate the 
    # outgoing field into s and p components. 

    reflection_s_incident = torch.empty(
        result_shape,
        dtype=torch.float64
    )

    reflection_p_incident = torch.empty(
        result_shape,
        dtype=torch.float64
    )

    transmission_s_incident = torch.empty(
        result_shape,
        dtype = torch.float64
    )

    transmission_p_incident = torch.empty(
        result_shape,
        dtype=torch.float64
    )

    for wavelength_index in range(number_of_wavelengths):
        # Use the already prepared first simulation we used
        # above to get the orders. Every wavelength needs a new S4
        # simulation because its material permittivities and 
        # frequency may be different.
        if wavelength_index ==0:
            prepared = first_prepared
        else:
            prepared = PreparedS4Model.from_model(
                model=model,
                config=config,
                wavelength_index=wavelength_index
            )
        
        current_orders = tuple(
            prepared.simulation.GetBasisSet()
        )

        for theta_index in range(number_of_thetas):
            theta_rad = float(
                theta_values[theta_index].item()
            )    
        
            for phi_index in range(number_of_phis):
                phi_rad = float(phi_values[phi_index].item())

                # Illumination with s polarised incident wave
                s_orders, reflected_s, transmitted_s = solve_s4_power_by_order(
                    prepared=prepared,
                    theta_rad=theta_rad,
                    phi_rad=phi_rad,
                    polarization="s"
                )

                # Illumination with a p polarised incident wave
                p_orders, reflected_p, transmitted_p = solve_s4_power_by_order(
                    prepared=prepared,
                    theta_rad = theta_rad,
                    phi_rad = phi_rad,
                    polarization = "p"
                )

                # The first three indices select this wavelength and 
                # angle combination. The final ':' means to store 
                # every diffraction order

                reflection_s_incident[
                    wavelength_index,
                    theta_index,
                    phi_index,
                    :
                ] = reflected_s

                reflection_p_incident[
                    wavelength_index,
                    theta_index,
                    phi_index,
                    :
                ] = reflected_p

                transmission_s_incident[
                    wavelength_index,
                    theta_index,
                    phi_index,
                    :
                ] = transmitted_s

                transmission_p_incident[
                    wavelength_index,
                    theta_index,
                    phi_index,
                    :,
                ] = transmitted_p
        
    return S4DiffractionResult(
        # Shape (Nw, Ntheta,Nphi,Norders)
        orders=orders,
        reflection_s_incident=reflection_s_incident,
        reflection_p_incident=reflection_p_incident,
        transmission_s_incident=transmission_s_incident,
        transmission_p_incident=transmission_p_incident
    )

def run_s4(
    model: Model,
    config: Config,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor
]:
    """Run S4 and return zeroth-order Rs, Rp, Ts, Tp.

    This is the public S4 backend that is to be used by the 
    benchmarking framework. It follows the same basic calling 
    convention as `run_metarcwa` and `run_fmmax`.

    Parameters
    ----------
    model:
        Solver-independent MetaRCWA model describing the physical
        system.
    config:
        Solver-independent numerical settings.
    
    Returns
    -------
    Rs:
        Zeroth-order reflected power for s-polarised incidence.
        Shape: (Nw, Ntheta, Nphi)
    Rp: 
        Zeroth-order reflected power for p-polarised incidence.
        Shape: (Nw, Ntheta, Nphi)
    Ts:
        Zeroth-order reflected power for s polarised incidence
        Shape: (Nw, Ntheta, Nphi)
    Tp:
        Zeroth-order transmitted power for p polarised incidence.
        Shape: (Nw, Ntheta, Nphi)
    
    Notes
    -----
    S4's per power power output doesn't seperate the outgoing
    field into s and p components. s and p describe the 
    polarisation of the incident wave. 

    For centred square at normal incidence example, symmetry
    makes cross polarised powers vanish so these values can be 
    compared with the MetaRCWA and FMMax outputs. 

    """

    # Keep the complete diffraction result available internally
    diffraction_result = run_s4_diffraction(
        model=model,
        config=config
    )

    # The standard benchmark interface currently compares specular
    # diffraction order (m,n) = (0,0)

    Rs, Rp, Ts, Tp = diffraction_result.powers_for_order(
        (0,0)
    )

    return Rs,Rp,Ts,Tp