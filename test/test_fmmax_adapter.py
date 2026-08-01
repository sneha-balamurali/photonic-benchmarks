import jax.numpy as jnp
from fmmax import basis, fmm

from src.fmmax.config import FMMaxConfig
from test.test_config import create_example_config
from test.test_metarcwa import create_small_config, create_small_model
from src.fmmax.simulation import prepare_fmmax_model

def test_fmmax_adapter() -> None:
    common = create_example_config()
    fmmax_config = FMMaxConfig.from_config(common)

    assert fmmax_config.dtype == jnp.float64
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

    model_spec, lattice_vectors, expansion = prepare_fmmax_model(
        model,
        config,
    )

    print("MetaRCWA lattice a1:", model_spec.a1)
    print("FMMax lattice u:", lattice_vectors.u)
    print("MetaRCWA lattice a2:", model_spec.a2)
    print("FMMax lattice v:", lattice_vectors.v)

    print("Requested terms:", (2 * config.m + 1) * (2 * config.n + 1))
    print("Actual FMMax terms:", expansion.num_terms)
    print("FMMax orders:")
    print(expansion.basis_coefficients)

    # The common float64 setting should survive conversion from PyTorch to JAX.
    assert lattice_vectors.u.dtype == jnp.float64
    assert lattice_vectors.v.dtype == jnp.float64

    # FMMax may adjust the requested number to keep its basis symmetric.
    assert expansion.num_terms > 0


if __name__ == "__main__":
    test_fmmax_adapter()
    test_fmmax_preparation()
    print("FMMax adapter and model preparation checks passed.")
