from test.test_metarcwa import create_small_config, create_small_model


def test_model_spec() -> None:
    """Inspect the numerical model that will be translated into FMMax."""

    model = create_small_model()
    config = create_small_config()

    model_spec = model.spec(
        nx=config.nx,
        ny=config.ny,
    )

    print(type(model_spec))
    print(model_spec)

    print("Number of finite layers:", len(model_spec.layers))
    print("Lattice vector a1:", model_spec.a1)
    print("Lattice vector a2:", model_spec.a2)
    print("Wavelength:", model_spec.wavelength)
    print("Incident permittivity:", model_spec.incidence.eps)
    print("Transmission permittivity:", model_spec.transmission.eps)

    for index, layer in enumerate(model_spec.layers):
        print(f"\nLayer {index}:")
        print("Type:", type(layer).__name__)
        print("Thickness:", layer.thickness)

        if hasattr(layer, "pattern"):
            print("Pattern shape:", layer.pattern.shape)
            print("Pattern minimum:", layer.pattern.min())
            print("Pattern maximum:", layer.pattern.max())
            print("Number of particle pixels:", layer.pattern.sum())


if __name__ == "__main__":
    test_model_spec()