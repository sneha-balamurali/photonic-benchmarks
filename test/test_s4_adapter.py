import math

from src.s4.config import S4Config
from test.test_config import create_example_config
from test.test_metarcwa import create_small_config, create_small_model
from src.s4.simulation import (
    PreparedS4Model,
    solve_s4_total_power,
    solve_s4_power_by_order,
    run_s4_diffraction,
    run_s4
)
import torch


def test_s4_config() -> None:
    """Check that the common configuration is translated for S4."""

    common = create_example_config()
    s4_config = S4Config.from_config(common)

    assert s4_config.nx == 32
    assert s4_config.ny == 32
    assert s4_config.requested_num_basis == 49
    assert s4_config.lattice_truncation == "Circular"


def test_s4_preparation() -> None:
    """Check that a MetaRCWA model can create an initial S4 simulation
    """

    model = create_small_model()
    config = create_small_config()

    prepared = PreparedS4Model.from_model(
        model=model,
        config=config
    )

    # create_small_model() currently contains one wavelength: 500 nm.
    assert prepared.wavelength_index == 0
    assert math.isclose(
        prepared.wavelength,
        500.0,
    )

    # The square-particle mode has a square lattice with 
    # period 180nm
    assert prepared.lattice_vectors == ((180.0,0.0),
                                        (0.0,180.0)
    )

    # m=2 and n=2 give the initial requested target:
    # (2m+1)(2n+1) = 5 * 5 = 25
    assert prepared.config.requested_num_basis == 25
    assert prepared.config.lattice_truncation == "Circular"

    # Calling an S4 method confirms that the returned 
    # object is a working S4 simulation and not just a
    # stored Python value
    reciprocal_a1, reciprocal_a2 = (
        prepared.simulation.GetReciprocalLattice()
    )
    
    # Relationship between real and reciprocal lattice vectors
    # are a1.b1 = 1
    assert math.isclose(
        reciprocal_a1[0],
        1.0 / 180.0
    )

    assert math.isclose(reciprocal_a1[1], 0.0)

    assert math.isclose(reciprocal_a2[0], 0.0)
    assert math.isclose(reciprocal_a2[1], 1.0 / 180.0)

    # GetBasisSet
    actual_basis = prepared.simulation.GetBasisSet()

    assert len(actual_basis) > 0
    assert (0,0) in actual_basis

    print("Requested S4 basis:", prepared.config.requested_num_basis)
    print("Actual S4 basis size:", len(actual_basis))
    print("Actual S4 orders:", actual_basis)

    # Check the material stack along z
    # z < 0: incidence medium
    # 0 < z < 20: homogeneous planarization layer
    # 20 < z < 100: patterned layer containing a solid square
    # within the void/background material
    # z > 100: transmission medium
    # Argument gives the coordinate at which to call the dielectric constant
    assert prepared.simulation.GetEpsilon(0,0,-1) == 1.0 + 0.0j
    assert prepared.simulation.GetEpsilon(0,0,10) == 2.25 + 0.0j

    # (0,0,50) would be inside the patterned layer because at the 
    # S4 unit cell centre, the point is inside the translated
    # solid square
    calculated_solid_eps = prepared.simulation.GetEpsilon(0,0,50)
    calculated_void_eps = prepared.simulation.GetEpsilon(60,60,50)
    expected_solid_eps = complex(prepared.model_spec.layers[1]
                                .medium_solid.eps[prepared.wavelength_index]
                                .detach()
                                .cpu()
                                .item())
    expected_void_eps = complex(prepared.model_spec.layers[1]
                                .medium_void.eps[prepared.wavelength_index]
                                .detach()
                                .cpu()
                                .item()
                                )

    # Because GetEpsilon() uses a finite Fourier reconstruction, 
    # compare which intended material each point is closer to than
    # expecting exact equality.

    assert abs(calculated_solid_eps - expected_solid_eps) < abs(
        calculated_solid_eps - expected_void_eps
    )

    assert abs(calculated_void_eps - expected_void_eps) < abs(
        calculated_void_eps - expected_solid_eps
    )

    print("Difference between the permittivty that S4's GetEpsilon"
            "reconstructrs using the finite retained Fourier basis and"
            "the given permittivty for the solid in the patterned" 
            f"layer is {calculated_solid_eps - expected_solid_eps}")



    transmission_eps = prepared.simulation.GetEpsilon(0,0,101)

    assert math.isclose(transmission_eps.real, -7.632, rel_tol = 1e-6)
    assert math.isclose(transmission_eps.imag, 0.731, rel_tol = 1e-6)

    expected_transmission_eps = complex(
                                        prepared.model_spec.transmission.eps[
                                        prepared.wavelength_index
                                        ].detach().cpu().item()
    )

    assert transmission_eps == expected_transmission_eps

def test_s4_total_power() -> None:
    """Check one normal-incidence, s-polarized S4 calculation."""

    model = create_small_model()
    config = create_small_config()

    prepared = PreparedS4Model.from_model(
        model=model,
        config=config,
        wavelength_index=0,
    )

    reflection, transmission = solve_s4_total_power(
        prepared=prepared,
        theta_rad=0.0,
        phi_rad=0.0,
        polarization="s",
    )

    print("S4 total s reflection:", reflection)
    print("S4 total s transmission:", transmission)

    assert math.isfinite(reflection)
    assert math.isfinite(transmission)

    assert reflection >= 0
    assert transmission >= 0

    # Run the same physical structure with a purely p-polarized
    # incident plane wave.
    reflection_p, transmission_p = solve_s4_total_power(
        prepared=prepared,
        theta_rad=0.0,
        phi_rad=0.0,
        polarization="p",
    )

    print("S4 total p reflection:", reflection_p)
    print("S4 total p transmission:", transmission_p)

    assert math.isfinite(reflection_p)
    assert math.isfinite(transmission_p)

    assert reflection_p >= 0
    assert transmission_p >= 0

    # A centred square on a square lattice is symmetric between the x
    # and y directions at normal incidence. The s and p results should
    # therefore agree within ordinary floating-point precision.
    assert math.isclose(
        reflection,
        reflection_p,
        rel_tol=1e-6,
    )

    assert math.isclose(
        transmission,
        transmission_p,
        rel_tol=1e-6,
    )
def test_s4_power_by_order() -> None:
    """Check that S4 separates power into retained diffraction orders."""

    model = create_small_model()
    config = create_small_config()

    prepared = PreparedS4Model.from_model(
        model=model,
        config=config,
        wavelength_index=0,
    )

    orders, reflection_by_order, transmission_by_order = (
        solve_s4_power_by_order(
            prepared=prepared,
            theta_rad=0.0,
            phi_rad=0.0,
            polarization="s",
        )
    )

    print("S4 diffraction orders:", orders)
    print("S4 reflection by order:", reflection_by_order)
    print("S4 transmission by order:", transmission_by_order)

    # There must be one reflected and transmitted power for every
    # retained S4 diffraction order.
    assert reflection_by_order.shape == (len(orders),)
    assert transmission_by_order.shape == (len(orders),)

    assert torch.isfinite(reflection_by_order).all()
    assert torch.isfinite(transmission_by_order).all()

    # Confirm that the physical zeroth order exists.
    zeroth_order_index = orders.index((0, 0))

    zeroth_reflection = reflection_by_order[zeroth_order_index]
    zeroth_transmission = transmission_by_order[zeroth_order_index]

    print("S4 zeroth-order reflection:", zeroth_reflection)
    print("S4 zeroth-order transmission:", zeroth_transmission)

    # Compare the sum of all individual orders with GetPowerFlux(),
    # which reports the total over all orders.
    total_reflection, total_transmission = solve_s4_total_power(
        prepared=prepared,
        theta_rad=0.0,
        phi_rad=0.0,
        polarization="s",
    )

    assert math.isclose(
        float(reflection_by_order.sum()),
        total_reflection,
        rel_tol=1e-9,
        abs_tol=1e-12,
    )

    assert math.isclose(
        float(transmission_by_order.sum()),
        total_transmission,
        rel_tol=1e-9,
        abs_tol=1e-12,
    )

def test_s4_diffraction_sweep() -> None:
    """Check that S4 creates the complete result tensor."""

    model = create_small_model()
    config = create_small_config()

    result = run_s4_diffraction(
        model=model,
        config=config,
    )

    number_of_orders = len(result.orders)

    # The small model has one wavelength, one theta and one phi.
    expected_shape = (
        1,
        1,
        1,
        number_of_orders,
    )

    assert result.reflection_s_incident.shape == expected_shape
    assert result.reflection_p_incident.shape == expected_shape
    assert result.transmission_s_incident.shape == expected_shape
    assert result.transmission_p_incident.shape == expected_shape

    assert torch.isfinite(
        result.reflection_s_incident
    ).all()

    assert torch.isfinite(
        result.reflection_p_incident
    ).all()

    assert torch.isfinite(
        result.transmission_s_incident
    ).all()

    assert torch.isfinite(
        result.transmission_p_incident
    ).all()

    Rs, Rp, Ts, Tp = result.powers_for_order(
        (0, 0)
    )

    print("S4 zeroth-order Rs:", Rs)
    print("S4 zeroth-order Rp:", Rp)
    print("S4 zeroth-order Ts:", Ts)
    print("S4 zeroth-order Tp:", Tp)

    assert Rs.shape == (1, 1, 1)
    assert Rp.shape == (1, 1, 1)
    assert Ts.shape == (1, 1, 1)
    assert Tp.shape == (1, 1, 1)    

def test_run_s4() -> None:
    """Check that the public S4 route returns Rs, Rp, Ts and Tp."""

    model = create_small_model()
    config = create_small_config()

    Rs, Rp, Ts, Tp = run_s4(
        model=model,
        config=config,
    )

    results = {
        "S4 Rs": Rs,
        "S4 Rp": Rp,
        "S4 Ts": Ts,
        "S4 Tp": Tp,
    }

    print(results)

    for value in results.values():
        assert value.shape == (1, 1, 1)
        assert torch.isfinite(value).all()

if __name__ == "__main__":
    test_s4_config()
    test_s4_preparation()
    test_s4_total_power()
    test_s4_power_by_order()
    test_s4_diffraction_sweep()
    test_run_s4()
    print("S4 adapter checks passed.")