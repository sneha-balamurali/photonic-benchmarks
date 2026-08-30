# Square-particle solver discrepancy

## Question

Why does S4 predict approximately 2% lower s-polarized
reflectance than FMMax and MetaRCWA for the
same square-particle structure?

## Baseline configuration

- Period: 180 nm
- Particle side: 60 nm
- Wavelength: 500 nm
- Normal incidence
- Material permittivities: ...
- Grid resolution: 96 × 96
- Circular Fourier truncation

[The complete YAML configuration](../../../src/configs/square_particle.yaml)

## Initial observation

FMMax and MetaRCWA agree to approximately at shared
actual Fourier-order counts. S4 remains approximately 2%
lower and appears to converge toward a different value.

![Rs convergence with fourier orders](../../../images/Rs_convergence_with_fourier_orders.png)

## Investigation plan

Following the project’s current scope, this discrepancy is documented here and deferred while the framework is generalised and factorisation methods are compared.

| Experiment | Question | Status | Conclusion |
|---|---|---|---|
| Circular basis comparison | Are the retained order sets identical? | In progress | — |
| Geometry resolution | Is the same square represented? | Planned | — |

## Current interpretation

The discrepancy does not appear to vanish by increasing the
number of Fourier orders. Geometry discretization might be a plausible explanation.