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

DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64
}

def build_media(data:dict,
                dtype: torch.dtype)-> dict[str, IsotropicMedium]:
    """
    Construct named MetaRCWA media from a dictionary of material data.
    """
    
    # Materials

    model_data = data["model"]
    material_data = model_data["materials"]
    
    # Store the constructed media under their material name
    media = {}
    
    for material_name in material_data:

        # Get the material's properties from the YAML file
        material_properties = material_data[material_name]

        # Get its epsilon dictionary
        epsilon = material_properties["epsilon"]

        # Extract the real and imaginary parts of epsilon
        eps_re = epsilon["real"]
        eps_im = epsilon["imag"]

        # Construct the DisperTorch material model for the material
        permittivity_model = ConstantEps(
            eps_re = eps_re,
            eps_im = eps_im,
            dtype=dtype
        )

        # Convert it into a MetaRCWA IstropicMedium
        medium = IsotropicMedium(from_dispertorch(permittivity_model))

        # Save it before moving onto the next material
        media[material_name] = medium

    # Return all media after the loop has finished
    return media

def build_layer(data:dict,
                dtype=torch.dtype
                )->Layer:

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
    dtype = DTYPE_MAP[model_data["numerics"]["dtype"]]
    media = build_media(data, dtype=dtype)

    model_data = data["model"]
    layer_data= model_data["layers"]

    layer_type = layer_data["type"]
    thickness_nm = float(layer_data["thickness_nm"])

    # Homogeneous layer
    if layer_type == "homogeneous":

        material = str(layer_data["material"])
        return Layer(
            medium_solid = media[material],
            thickness=thickness_nm
        )

    # A patterned layer contains a shape material surrounded by 
    # a background material

    if layer_type == "patterned":

        solid = str(layer_data["solid"])
        void = str(layer_data["void"])
        shape_data = layer_data["shape"]
        # MetShapes constructs the shape from its dictionary
        shape = Shape.from_parametric(shape_data)

        return Layer(
            medium_solid = media[solid],
            medium_void = media[void],
            thickness=torch.tensor(thickness_nm),
            shape_fn = from_metashapes(shape, soft=False)
        )


def build_model(data:dict,
                dtype=torch.dtype
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
    dtype = DTYPE_MAP[model_data["numerics"]["dtype"]]
    media = build_media(data, dtype=dtype)

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

    source = Source(wavelength=wavelength,
                    theta=torch.deg2rad(theta_deg), 
                    phi=torch.deg2rad(phi_deg))

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
        incidence=media["incidence"],
        layers = layers,
        transmission=media["transmission"],
        lattice = lattice
    )

    return Model(stack,source)