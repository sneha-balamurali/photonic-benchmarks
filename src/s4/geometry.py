"""Shape conversions from Metashapes geometry to S4 compatible

MetaRCWA uses MetShapes geometry to produce raster masks for
grid based solvers. S4 instead expects analytical region commands
such as SetRegionRectangle().

This module contains the mapping between those two geometry descriptions.
Only geometry types listed in _SHAPE_ADAPTERS are supported.
"""

from typing import Any

from metarcwa.model.layer import Layer  
from metashapes import Shape
from metashapes.shape import Rectangle

def metashape_from_layer(layer: Layer) -> Shape:
    """Return the MetaShapes geometry registered on a MetaRCWA layer

    A patterned layer created with from_metashapes(shape) contains a
    CallableModule. MetaRCWA registers the original MetaShapes as a 
    dependency of that CallableModule so that PyTorch can track its parameters.

    This initial S4 adapter supports only that standard from_metashapes(shape) 
    route. 

    Parameters
    ----------
    layer:
        An unresolved MetaRCWA layer taken from model.stack.layers.

    Returns
    --------
    Shape:
        The original top-level MetaShapes geometry.

    Raises
    ------
    ValueError:
        If the supplied layer is homogeneous and therefore has no shape.
    TypeError:
        If the shape function was not created in the expected way.
    
    """

    if layer.shape_fn is None:
        raise ValueError(
            "A homogeneous layer does not contain analytical geometry"
        )
    
    #getattr() lets us provide an error if this is a custom
    # shape function that doesn't have MetRCWA's _dep attribute
    dependencies = getattr(layer.shape_fn, "_deps", ())

    if len(dependencies) !=1:
        raise TypeError(
            "Expected one MetShapes object registered through"
            "from_metashapes(shape) but instead found"
            f"{len(dependencies)} dependecies"
        )
    
    shape = dependencies[0]

    # A dependency could also be some other PyTorch Model,
    # so verify that it really is a MetShapes Shape

    if not isinstance(shape,Shape):
        raise TypeError(
            "The registered dependency is not a MetaShapes Shape"
            f"Recieved {type(shape).__name__!r}."
        )
    
    return shape

def add_rectangle(
    simulation: Any,
    *,
    layer_name: str,
    material_name: str,
    shape: Rectangle,
) -> None:
    """Translate one MetaShapes Rectangle into an S4 region.

    MetShapes stores the complete rectangle width and height 
    in `size`. S4 instead expects `Halfwifth`, so both dimensions
    have to be divided by 2.

    MetaShapes supports rounded rectangle corders but S4's
    `SetRegionRectangle()` doesn't. A nonzero corner radius is
    rejected so the adapter can't silently change the intended
    geometry. 

    Parameters
    ----------
    simulation:
        The S4 simulation object returned by S4.New()
    layer_name: 
        Name of the existing S4 layer recieving the rectangle.
    material_name:
        Name of the existing S4 solid material.
    shape:
        The MetaShapes Rectangle being translated.
    """

    parameters = shape.to_parametric()

    # These are the MetaShapes Rectangle parameters that this adapter
    # knows how to translate or validate

    center = parameters["center"]
    size = parameters["size"]
    angle = parameters["angle"]
    corner_radius= parameters["corner_radius"]

    if corner_radius !=0:
        raise NotImplementedError(
            "S4 rectangles do not support rounded corners."
        )
    
    # MetaShapes coordinates are relative to the unit cell origin
    # S4 coordinates are relative to the unit-cell center
    halfwidths = (
        size[0] / 2, 
        size[1] / 2,
    )

    simulation.SetRegionRectangle(
        Layer=layer_name,
        Material=material_name,
        Center = s4_center,
        Angle=angle,
        Halfwidths=halfwidths,
    )

# Mapping dictionary
# Key: The exact MetaShapes class that we support
# Value: The function that knows how to translate that class into S4
# Currently only Rectangle is supported
_SHAPE_ADAPTERS = {
    Rectangle: add_rectangle,
}

def add_shape(
    simulation: Any,
    *,
    layer_name: str,
    material_name: str,
    shape: Shape,
) -> None:
    """Add one supported MetaShapes geometry to an S4 layer.

    The shape's exact Python class is looked up in _SHAPE_ADAPTERS.
    Unsupported classes are rejected instead of being guessed.
    """
    # For a rectange, this lookup would return add_rectangle
    adapter = _SHAPE_ADAPTERS.get(type(shape))

    # Dictionary.get() returns None when the key is not present
    if adapter is None:
        shape_name = type(shape).__name__

        raise NotImplementedError(
            "The S4 backend does not yet support the MetaShape"
            "Currently only Rectangle is supported."
        )

    # Call the translation function selected by the dictionary.
    # For the current square model, this calls add_rectangle(...)
    adapter(
        simulation,
        layer_name=layer_name,
        material_name=material_name,
        shape=shape
    )