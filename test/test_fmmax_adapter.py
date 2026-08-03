import jax.numpy as jnp
from fmmax import basis, fmm

from src.fmmax.config import FMMaxConfig
from test.test_config import create_example_config
from test.test_metarcwa import create_small_config, create_small_model
from src.fmmax.simulation import PreparedFMMaxModel

def test_fmmax_adapter() -> None:
    common = create_example_config()
    fmmax_config = FMMaxConfig.from_config(common)

    assert fmmax_config.real_dtype == jnp.float64
    assert fmmax_config.complex_dtype == jnp.complex128
    assert fmmax_config.device == "cpu"
    assert fmmax_config.nx == 32
    assert fmmax_config.ny == 32
    assert fmmax_config.approximate_num_terms == 49
    assert fmmax_config.truncation is basis.Truncation.CIRCULAR
    assert fmmax_config.formulation is fmm.Formulation.FFT

def test_fmmax_preparation() -> None:
    """Check that the model lattice and basis can be prepared for FMMax."""

    model = create_small_model()
    config = create_small_config()

    prepared = PreparedFMMaxModel.from_model(
    model,
    config,
    )

    print("MetaRCWA lattice a1:", prepared.model_spec.a1)
    print("FMMax lattice u:", prepared.lattice_vectors.u)
    print("MetaRCWA lattice a2:", prepared.model_spec.a2)
    print("FMMax lattice v:", prepared.lattice_vectors.v)

    print(
        "Requested terms:",
        (2 * config.m + 1) * (2 * config.n + 1),
    )
    print("Actual FMMax terms:", prepared.expansion.num_terms)
    print("FMMax orders:")
    print(prepared.expansion.basis_coefficients)

    assert prepared.lattice_vectors.u.dtype == jnp.float64
    assert prepared.lattice_vectors.v.dtype == jnp.float64

    assert prepared.lattice_vectors.u.shape == (2,)
    assert prepared.lattice_vectors.v.shape == (2,)

    assert prepared.wavelength.shape == (1, 1, 1)
    assert prepared.in_plane_wavevector.shape == (1, 1, 1, 2)

    assert len(prepared.permittivities) == 4
    assert len(prepared.thicknesses) == 4
    assert len(prepared.layer_solve_results) == 4

    assert prepared.permittivities[0].shape == (1, 1, 1, 1, 1)
    assert prepared.permittivities[1].shape == (1, 1, 1, 1, 1)
    assert prepared.permittivities[2].shape == (1, 1, 1, 32, 32)
    assert prepared.permittivities[3].shape == (1, 1, 1, 1, 1)

    assert prepared.expansion.num_terms > 0
    assert prepared.s_matrix is not None


if __name__ == "__main__":
    test_fmmax_adapter()
    test_fmmax_preparation()
    print("FMMax adapter and model preparation checks passed.")
