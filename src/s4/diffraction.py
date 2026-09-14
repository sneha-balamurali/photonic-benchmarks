from s4.simulation import PreparedS4Model
import torch
import math
from dataclasses import dataclass


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