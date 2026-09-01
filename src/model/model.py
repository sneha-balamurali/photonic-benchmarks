from pathlib import Path

import torch
import yaml

from metarcwa import (Source, 
                      Lattice, 
                      IsotropicMedium, 
                      Layer, 
                      Stack, 
                      Model
)

from metashapes import Shape
from metashapes.shape import Rectangle, Ellipse

from dispertorch import ConstantEps
from metarcwa.model.adapters import (from_dispertorch, 
                                     from_metashapes
)

def build_model(
        lattice_a1: torch.Tensor,
        lattice_a2: torch.Tensor,
        wavelength: torch.Tensor,
        theta_deg: torch.Tensor,
        phi_deg: torch.Tensor,
        incidence_perm_model: ConstantEps,
        dielectric_perm_model: ConstantEps,
        particle_perm_model: ConstantEps,
        transmission_perm_model: ConstantEps,
        layers_data: list[dict]
) -> Model:
    """
    Construct a solver independent MetaRCWA model.

    The lengths are given in nanometers. Source angles are 
    given in degrees and converted to radians for MetaRCWA.

    The complete layer stack from incidence side to the transmission side is:
    1. Semi-infinite incidence medium
    2. Homogeneous dielectric layer
    3. Patterned layer containing the particle and its background
    4. Semi-infinite transmission medium

    Parameters:
    lattice_a1_nm: torch.Tensor
        Lattice vector a1 in nanometers.
    lattice_a2_nm: torch.Tensor
        Lattice vector a2 in nanometers.
    wavelength: torch.Tensor   
        Free-space illumination wavelength(s) in nanometers.
    theta_deg: torch.Tensor
        Polar incidence angle(s) in degrees, measured from the surface normal. Use zero
        for normal incidence.
    phi_deg: torch.Tensor
        Azimuthal incidence angle(s) in degrees.
    incidence_perm_model: ConstantEps
        DispertTorch material model for the semi-infinite incidence medium.
    dielectric_perm_model: ConstantEps
        DispertTorch material model for the homogeneous dielectric layer.
    particle_perm_model: ConstantEps
        DispertTorch material model for the particle in the patterned layer.
    transmission_perm_model: ConstantEps
        DispertTorch material model for the semi-infinite transmission medium.
    layers_data: list[dict]
        List of dictionaries containing the layer data. Each dictionary should have the following keys:
        - 'thickness_nm': Thickness of the layer in nanometers.
        - 'shape': Shape of the particle in the patterned layer (e.g., 'rectangle', 'ellipse').
        - 'shape_params': Parameters for the shape (e.g., side lengths for rectangle, axes for ellipse).

    Returns:
    --------
    Model:
        A solver independent MetaRCWA model.
    """

def build_layer(data:dict)->Layer:
    """
    Construct a MetaRCWA layer from a dictionary of layer data.

    Parameters:
    -----------
    data: dict
        Dictionary containing the layer data. It should have the following keys:
        - 'thickness_nm': Thickness of the layer in nanometers.
        - 'shape': Shape of the particle in the patterned layer (e.g., 'rectangle', 'ellipse').
        - 'shape_params': Parameters for the shape (e.g., side lengths for rectangle, axes for ellipse).
    Returns:
    --------
    Layer:
        A MetaRCWA Layer object.
        """

    model_data = data["model"]
    layer_data= model_data["layers"]

    layer_type = layer_data["type"]
    thickness_nm = float(layer_data["thickness_nm"])

    # Homogeneous layer
    if layer_type == "homogeneous":

        return Layer(
            medium_solid = from_dispertorch(layer_data["material"]),
            thickness=thickness_nm
        )

    # A patterned layer contains a shape material surrounded by 
    # a background material

    if layer_type == "patterned":

        shape_data = layer_data["shape"]
        # MetShapes constructs the shape from its dictionary
        shape = Shape.from_parametric(shape_data)

        return Layer(
            medium_solid = IsotropicMedium(
                from_dispertorch(layer_data["solid"])),
            medium_void = IsotropicMedium(
                from_dispertorch(layer_data["void"])),
            thickness=torch.tensor(thickness_nm),
            shape_fn = from_metashapes(shape, soft=False)
        )


def build_model(data:dict) -> Model:
    """
    Construct a solver independent MetaRCWA model.

    The lengths are given in nanometers. Source angles are 
    given in degrees and converted to radians for MetaRCWA.

    The complete layer stack from incidence side to the transmission side is:
    1. Semi-infinite incidence medium
    2. Homogeneous dielectric layer
    3. Patterned layer containing the particle and its background
    4. Semi-infinite transmission medium

    Parameters:
    lattice_a1_nm: torch.Tensor
        Lattice vector a1 in nanometers.
    lattice_a2_nm: torch.Tensor
        Lattice vector a2 in nanometers.
    wavelength: torch.Tensor   
        Free-space illumination wavelength(s) in nanometers.
    theta_deg: torch.Tensor
        Polar incidence angle(s) in degrees, measured from the surface normal. Use zero
        for normal incidence.
    phi_deg: torch.Tensor
        Azimuthal incidence angle(s) in degrees.
    incidence_perm_model: ConstantEps
        DispertTorch material model for the semi-infinite incidence medium.
    dielectric_perm_model: ConstantEps
        DispertTorch material model for the homogeneous dielectric layer.
    particle_perm_model: ConstantEps
        DispertTorch material model for the particle in the patterned layer.
    transmission_perm_model: ConstantEps
        DispertTorch material model for the semi-infinite transmission medium.
    layers_data: list[dict]
        List of dictionaries containing the layer data. Each dictionary should have the following keys:
        - 'thickness_nm': Thickness of the layer in nanometers.
        - 'shape': Shape of the particle in the patterned layer (e.g., 'rectangle', 'ellipse').
        - 'shape_params': Parameters for the shape (e.g., side lengths for rectangle, axes for ellipse).

    Returns:
    --------
    Model:
        A solver independent MetaRCWA model.
    """

    # YAML has nested dictionaries so index each level seperately
    model_data = data["model"]
    lattice_data = model_data["lattice"]
    source_data = model_data["source"]

    
    # Extract the parameters from the data dictionary

    # Lattice
    lattice_a1 = torch.tensor(lattice_data["a1_nm"])
    lattice_a2 = torch.tensor(lattice_data["a2_nm"])
    lattice = Lattice(a1=lattice_a1, a2=lattice_a2)

    # Source
    wavelength = torch.tensor(source_data["wavelength_nm"])
    theta_deg = torch.tensor(source_data["theta_deg"])
    phi_deg = torch.tensor(source_data["phi_deg"])

    source = Source(wavelength=wavelength, theta=theta_deg, phi=phi_deg)

    # Materials
    incidence_perm_model = ConstantEps(torch.tensor(model_data["incidence_perm"]))
    dielectric_perm_model = ConstantEps(torch.tensor(model_data["dielectric_perm"]))
    particle_perm_model = ConstantEps(torch.tensor(model_data["particle_perm"]))
    transmission_perm_model = ConstantEps(torch.tensor(model_data["transmission_perm"]))

    # Convert the DisperTorch material models into MetaRCWA media
    # The other layers are constructed from the build_layer function defined above
    incidence_medium = IsotropicMedium(from_dispertorch(incidence_perm_model))
    transmission_medium = IsotropicMedium(from_dispertorch(transmission_perm_model))

    # Layers
    layers_data = model_data["layers"]

    # Empty list of constructed MetaRCWA layers
    layers = []

    # Read every layer listed in the YAML file
    for layer_data in layers_data:
        layer = build_layer(layer_data)
        # Add to stack in same order as YAML
        layers.append(layer)

    # Construct the stack
    stack = Stack(
        incidence=incidence_medium,
        layers = layers,
        transmission=transmission_medium,
        lattice = lattice
    )

    return Model(stack,source)