import torch
# Make newly created floating point tensors use float64 by default
torch.set_default_dtype(torch.float64)

from dispertorch import ConstantEps

from metashapes.shape import Rectangle

from metarcwa import (
    Lattice,
    Layer,
    IsotropicMedium,
    Model,
    Source,
    Stack
)

from metarcwa.model.adapters import (
    from_dispertorch,
    from_metashapes
)

# Define the square lattice primitive vectors
a1_nm = torch.tensor([180.0,0.0])
a2_nm = torch.tensor([0.0,180.0])

lattice = Lattice(
    a1 = a1_nm,
    a2 = a2_nm
)

# MetaRCWA IsotropicMedium expects a callable permittivity model.
# DisperTorch is a material library that calculates a material's
# permittivity at the requested wavelength in the required callable form.
# Using DisperTorch also enables wavelength-dependent material models to be
# substituted later without changing the rest of the model construction.
# ConstantEps defines its own constructors where it defaults to float32 
# so will re-specify

incidence_perm_model = ConstantEps(
    eps_re = 1.0,
    eps_im = 0.0,
    dtype = torch.float64
)

planarization_perm_model = ConstantEps(
    eps_re = 2.25,
    eps_im = 0.0,
    dtype=torch.float64
)

particle_perm_model = ConstantEps(
    eps_re = 4.0,
    eps_im = 0.0,
    dtype = torch.float64
)

pattern_background_perm_model = ConstantEps(
    eps_re = 2.25,
    eps_im = 0.0,
    dtype=torch.float64
)

transmission_perm_model = ConstantEps(
    eps_re = 4.0,
    eps_im = 0.0,
    dtype=torch.float64
)

# Convert the DisperTorch Material Models into 
# MetaRCWA media

incidence_medium = IsotropicMedium(
    from_dispertorch(incidence_perm_model)
)

planarization_medium = IsotropicMedium(
    from_dispertorch(planarization_perm_model)
)

particle_medium = IsotropicMedium(
    from_dispertorch(particle_perm_model
    )
)

pattern_background_medium = IsotropicMedium(
    from_dispertorch(pattern_background_perm_model)
)

transmission_medium = IsotropicMedium(
    from_dispertorch(transmission_perm_model)
)

# Define particle
# Metashapes uses coordinates within [0,period]
square_side_length = 60.0
square_center = torch.tensor([a1_nm[0]/2, a2_nm[1]/2])
size = torch.tensor([square_side_length,square_side_length])
angle=torch.tensor(0.0)
corner_radius = 0.0
square_geometry = Rectangle(center=square_center,
                            size=size,
                            angle=angle,
                            corner_radius=corner_radius)

# Define finite layers
planarization_thickness = 20.0
planarization_layer = Layer(medium_solid = planarization_medium,
                            thickness = planarization_thickness)

patterned_layer_thickness = 80.0
patterned_layer = Layer(medium_solid = particle_medium,
                        thickness = patterned_layer_thickness,
                        medium_void = pattern_background_medium,
                        shape_fn = from_metashapes(square_geometry))


# Stack the layers
stack = Stack(incidence = incidence_medium,
              layers = [planarization_layer, patterned_layer],
              transmission = transmission_medium,
              lattice = lattice)

# Define the source
wavelength_nm = 500.0
theta_rad = 0.0
phi_rad = 0.0

source = Source(wavelength=wavelength_nm,
                theta = theta_rad,
                phi = phi_rad)

model = Model(stack,source)