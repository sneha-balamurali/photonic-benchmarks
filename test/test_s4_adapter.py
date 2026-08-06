from src.s4.config import S4Config
from test.test_config import create_example_config


def test_s4_config() -> None:
    """Check that the common configuration is translated for S4."""

    common = create_example_config()
    s4_config = S4Config.from_config(common)

    assert s4_config.nx == 32
    assert s4_config.ny == 32
    assert s4_config.requested_num_basis == 49
    assert s4_config.lattice_truncation == "Circular"


if __name__ == "__main__":
    test_s4_config()
    print("S4 configuration check passed.")