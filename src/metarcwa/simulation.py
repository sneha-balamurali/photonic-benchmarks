import torch
from metarcwa import Model, Solver, Observables
from copy import deepcopy
from src.metarcwa.config import MetaRCWAConfig

def run_metarcwa(model, config) -> dict:
    """
    Return the total reflected and transmitted power for the
    s and p incidence.

    Rs,Rp,Ts,Tp have shape:
    [N_wavelength, N_theta, Nphi]

    Illumination is a plane wave in incident order (0,0)
    Powers include both output polarisations and all retained
    orders.

    This power conversion requires lossless, positive permittivity
    incidence and transmission media. 
    The finite layers in between may absorb.
    """
    # Translate the shared numerical settings
    cfg = MetaRCWAConfig.from_config(config)

    # Run the solver 
    # Solver changes the model's dtype/device
    # copy it to preserve the shared model
    
    # Prepares the EM modes for the layers
    solver = Solver(deepcopy(model),cfg)
    # Combines the layers into the scattering matrix
    solution = solver.run()
    # Provides access to the reflected and 
    # transmitted E field amplitudes
    obs = Observables(solution)

    # Get the retained diffraction orders
    # [[-1,0],[0,-1]...] i.e. each row is one [m,n] pair
    # shape: [N_orders,2]
    ls = solver.layersolver
    orders = torch.stack(
        (ls.m_flat, ls.n_flat),
        dim=-1
    )

    # Material values already evaluated at every source wavelength
    spec = solver.model_spec

    # Get the incidence and transmission permittivities
    # permittivity shape: [Nw, Ntheta, Nphi, Norders]
    # singleton dimensions lets us reuse the wavelength 
    # permittivity across every angle and order

    eps_inc = spec.incidence.eps.reshape(-1,1,1,1)
    eps_tran = spec.transmission.eps.reshape(-1,1,1,1)

    # Transverse wavevectors for every wavelength, angle, order
    # MetaRCWA normalises these by the free space wavenumber,
    # k0 = 2pi / wavelength
    kx = ls.kx
    ky=ls.ky

    # kz^2 = epsilon - kx^2 - ky^2
    # In lossless outer media, evanescent orders have Re(z) = 0
    kz_inc = torch.sqrt(eps_inc - kx**2-ky**2+0j).real
    kz_tran = torch.sqrt(eps_tran - kx**2-ky**2+0j).real

    # Locate the incidence order (0,0) without assuming its array position
    order_pairs = [tuple(pair) for pair in orders.tolist()]
    zero_index = order_pairs.index((0,0))

    # kz_inc contains the incidence medium wavevectors for every
    # retained diffraction order. But in our illumination we are
    # only illuminating with one plane wave, labelled (0,0)
    # Need to normalise with the power of that wave
    incident_weight = kz_inc[...,zero_index]

    results = {}

    # Two seperate illumination experiments: s incidence and p incidence
    for polarization in ("s","p"):
        reflectance = torch.zeros_like(incident_weight)
        transmittance = torch.zeros_like(incident_weight)

        # Add the outgoing power from every retained diffraction order
        for index, output_order in enumerate(order_pairs):
            rs,rp = obs.reflection(
                pol = polarization,
                in_order = (0,0),
                out_order = output_order
            )

            ts,tp = obs.transmission(
                pol=polarization,
                in_order=(0,0),
                out_order = output_order
            )

            # Include both output polarisations
            # Equation: |rs_j|² + |rp_j|²
            reflected_amplitude_squared = rs.abs()**2 + rp.abs()**2
            # Equation: |ts_j|² + |tp_j|²
            transmitted_amplitude_squared = ts.abs()**2+tp.abs()**2

            # Convert amplitude squared into a fraction of incident power
            # i.e. Reflectance and Transmittance
            reflectance += (
                kz_inc[...,index]
                / incident_weight
                * reflected_amplitude_squared
            )

            transmittance += (
                kz_tran[...,index]
                / incident_weight
                * transmitted_amplitude_squared
            )

        # Each result has shape (N_wavelength, Ntheta, Nphi)
        results["R" + polarization] = reflectance
        results["T" + polarization] = transmittance

    # Shared bookkeeping convention
    results["requested_orders"] = (
        (2*cfg.m + 1)* (2*cfg.n+1)
    )
    # Number of retained orders
    results["actual_orders"] = len(order_pairs)
    #The retained [m,n] pairs
    results["basis_coefficients"] = orders

    return results

        
