import torch

from src.config import Config
from metarcwa.config_prev import MetaRCWAConfig


def create_example_config() -> Config:
    """Create one small configuration used by these tests."""

    return Config(
        dtype="float64",
        device="cpu",
        nx=32,
        ny=32,
        m=3,
        n=3,
        truncation="circular",
    )


def test_config_round_trip() -> None:
    """Check that the common configuration can be
    converted into a dictionary and reconstructed
    back without changing."""

    original = create_example_config()

    dictionary = original.to_dict()
    restored = Config.from_dict(dictionary)

    assert restored == original


def test_metarcwa_adapter() -> None:
    """Check that the MetaRCWA adapter can correctly
    translate the common Config into a valid
    MetaRCWA configuration."""

    common = create_example_config()
    metarcwa_config = MetaRCWAConfig.from_config(common)

    assert metarcwa_config.dtype == torch.float64
    assert str(metarcwa_config.device) == "cpu"
    assert metarcwa_config.nx == common.nx
    assert metarcwa_config.ny == common.ny
    assert metarcwa_config.m == common.m
    assert metarcwa_config.n == common.n
    assert metarcwa_config.truncation == common.truncation


if __name__ == "__main__":
    test_config_round_trip()
    test_metarcwa_adapter()

    print("Both configuration checks passed.")