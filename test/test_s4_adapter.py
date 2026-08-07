import math

from src.s4.config import S4Config
from src.s4.simulation import PreparedS4Model
from test.test_config import create_example_config
from test.test_metarcwa import create_small_config, create_small_model


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


if __name__ == "__main__":
    test_s4_config()
    test_s4_preparation()
    print("S4 adapter checks passed.")