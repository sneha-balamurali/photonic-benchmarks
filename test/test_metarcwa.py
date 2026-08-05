import torch
from dispertorch import ConstantEps

from examples.metarcwa.square_particle import build_model
from src.config import Config
from src.metarcwa.simulation import run_metarcwa


def create_small_model():
    """Create a square-particle model for a first solver test."""

    return build_model(
        particle_side_length_nm=torch.tensor(60.0),
        planarization_layer_thickness_nm=torch.tensor(20.0),
        patterned_layer_thickness_nm=torch.tensor(80.0),
        lattice_period_nm=torch.tensor(180.0),
        wavelength_nm=torch.tensor([500.0]),
        theta_rad=torch.tensor([0.0]),
        phi_rad=torch.tensor([0.0]),
        incidence_perm_model=ConstantEps(1.0),
        planarization_perm_model=ConstantEps(2.25),
        particle_perm_model=ConstantEps(eps_re=-7.632, eps_im=0.731),
        pattern_background_perm_model=ConstantEps(2.25),
        transmission_perm_model=ConstantEps(eps_re=-7.632, eps_im=0.731),
    )

def create_small_config() -> Config:
    """Create a low-resolution configuration so the test runs quickly."""

    return Config(
        dtype="float64",
        device="cpu",
        nx=32,
        ny=32,
        m=2,
        n=2,
        truncation="circular",
    )


def test_metarcwa_simulation() -> None:
    """Check that the complete MetaRCWA route returns finite results."""

    model = create_small_model()
    config = create_small_config()

    Rs, Rp, Ts, Tp = run_metarcwa(model, config)

    results = {
        "Metarcwa Rs": Rs,
        "Metarcwa Rp": Rp,
        "Metarcwa Ts": Ts,
        "Metarcwa Tp": Tp,
    }
    
    # Check for any NaN values
    for value in results.values():
        assert torch.isfinite(value).all()

    print(results)


if __name__ == "__main__":
    test_metarcwa_simulation()
    print("MetaRCWA simulation check passed.")