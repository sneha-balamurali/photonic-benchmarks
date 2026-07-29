# Square particle using only MetaRCWA model classes

import torch

from metarcwa import Source, Lattice, IsotropicMedium, Layer, Stack, Model
from metashapes.shape import Rectangle

from dispertorch import ConstantEps
from metarcwa.model.adapters import from_dispertorch, from_metashapes

def build_square_particle_model(particle_side_length_nm: torch.Tensor,
                planarization_layer_thickness_nm: torch.Tensor,
                patterned_layer_thickness_nm: torch.Tensor,
                lattice_period_nm: torch.Tensor,
                wavelength_nm: torch.Tensor,
                theta_rad: torch.Tensor, 
                phi_rad: torch.Tensor,
                incidence_perm_model: ConstantEps,
                planarization_perm_model: ConstantEps,
                particle_perm_model: ConstantEps,
                pattern_background_perm_model: ConstantEps,
                transmission_perm_model: ConstantEps,
                ) -> Model:
    """
        Construct a periodic square-particle MetaRCWA model.

    The complete layer stack from incidence side to the transmission side is:
    1. Semi-infinite incidence medium
    2. Homogeneous planarization layer
    3. Patterned layer containing the square particle and its background
    which in this case is the same material as the planarization layer
    4. Semi-infinite transmission medium

    This function only constructs the solver-independent physical model. It doesn't run
    MetaRCWA, S4 or FMMax.

    Parameters:
    -----------
    particle_side_length_nm: torch.Tensor
        Side length of the square particle in nanometers. The same length is used in 
        both the x and y directions.
    planarization_layer_thickness_nm: torch.Tensor
        Thickness of the planarization layer above the patterned layer and below the
        incidence medium in nanometers.
    patterned_layer_thickness_nm: torch.Tensor
        Thickness of the layer containing the square particle, in nanometers.
    lattice_period_nm: torch.Tensor
        Period of the square lattice in nanometers. It is the same in the x and y 
        directions.
    wavelength_nm: torch.Tensor
        Free-space illumination wavelength(s) in nanometers.
    theta_rad: torch.Tensor
        Polar incidence angle(s) in radians, measured from the surface normal. Use zero
        for normal incidence.
    phi_rad: torch.Tensor
        Azimuthal incidence angle(s) in radians
    incidence_perm_model: ConstantEps
        DispertTorch material model for the semi-infinite incidence medium
    planarization_perm_model: ConstantEps
        DisperTorch material model for the homogeneous planarization layer
    particle_perm_model: ConstantEps
        DisperTorch material model for the material inside the square. This material occupies
        the region where the geometry mask equal to one.
    pattern_background_perm_model: ConstantEps
        DisperTorch material model surrounding the particle inside the patterned layer.
        This material occupies the region where the geometry mask equals zero.
    transmission_perm_model: ConstantEps
        DisperTorch material model for the semi-infinite substrate/transmission material.

    Returns
    -------
    Model:
        A MetaRCWA model containing the completed stack and source. The model can be
        resolved into a ModelSpec and translated by S4 or FMMax backend. 

    """
    # Define lattice vectors
    lattice = Lattice.rectangular(px = lattice_period_nm, py=lattice_period_nm)

    # Define materials
    incidence_medium = IsotropicMedium(from_dispertorch(incidence_perm_model))
    planarization_medium = IsotropicMedium(from_dispertorch(planarization_perm_model))
    particle_medium = IsotropicMedium(from_dispertorch(particle_perm_model))
    pattern_background_medium = IsotropicMedium(from_dispertorch(pattern_background_perm_model))
    transmission_medium = IsotropicMedium(from_dispertorch(transmission_perm_model))

    # Define square geometry
    center = torch.tensor([lattice_period_nm / 2, lattice_period_nm / 2])
    size = torch.tensor([particle_side_length_nm,particle_side_length_nm])
    angle = torch.tensor(0.0)
    corner_radius= torch.tensor(0.0)
    square_geometry = Rectangle(center = center,
                                size = size,
                                angle = angle,
                                corner_radius = corner_radius,
                                )

    # Define finite Layers
    planarization_layer = Layer(medium_solid = planarization_medium, 
                                thickness = planarization_layer_thickness_nm,
                                )
    patterned_layer = Layer(medium_solid = particle_medium, 
                            thickness = patterned_layer_thickness_nm,
                            medium_void = pattern_background_medium, 
                            shape_fn = from_metashapes(square_geometry),
                            )

    # Stack the layers
    stack = Stack(incidence = incidence_medium,
                 layers = [planarization_layer, patterned_layer],
                 transmission = transmission_medium, 
                 lattice = lattice,
                 )

    # Define the source
    source = Source(wavelength = wavelength_nm, 
                    theta = theta_rad,
                    phi = phi_rad,
                    )

    return Model(stack=stack,
                source = source,
                )