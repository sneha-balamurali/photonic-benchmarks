import jax.numpy as jnp
from fmmax import basis, fmm

from src.fmmax.config import FMMaxConfig
from test.test_config import create_example_config

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

if __name__ == "__main__":
    test_fmmax_adapter()

    print("FMMax adapter check passed.")