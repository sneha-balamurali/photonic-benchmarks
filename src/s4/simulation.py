from dataclasses import dataclass
from typing import Any

import S4
import torch
from metarcwa import Model

from src.config import Config
from src.s4.config import S4Config

from metarcwa.model.layer import HomogeneousLayer, PatternedLayer

def tensor_to_tuple(value:torch.Tensor) -> tuple[float,float]:
    """Convert a two-component PyTorch tensor into an ordinary
    Python tuple.

    MetaRCWA stores each lattice vector as a PyTorch tensor 
    with shape (2,) but S4's python interface expects 
    something like ((a1_x,a1_y), (a2_x,a2y))

    Gradient tracking and GPU storage are not meaningful to S4, 
    so the tensor is detached and moved to CPU before extracting
    its values.
    """

    # Check that S4 lattice vectors only contains 2 componenets
    if value.numel() != 2:
        raise ValueError(
            "A S4 lattice vector must contain exactly 2 components."
            f"Recieved shape {tuple(value.shape)}."
        )

    flattened = value.detach().cpu().reshape(-1)

    return (float(flattened[0]),float(flattened[1]))

def tensor_to_float(value: torch.Tensor) -> float:
    """Convert a one-value PyTorch tensor into a Python float."""

    if value.numel() != 1:
        raise ValueError(
            "Expected a tensor containing exactly one value. "
            f"Received shape {tuple(value.shape)}."
        )

    return float(value.detach().cpu().item())

def permittivity_at_wavelength(
    value: torch.Tensor,
    wavelength_index: int
) -> complex:
    """Select one wavelength's permittivity as a Python complex number.

    MetaRCWA evaluates an isotropic material at every requested wavelength,
    producing a tensor with shape (Nw,). S4 has one active material value at 
    a time, so we select the value for one wavelength.

    Parameters:
    -----------
    value:
        Complex MetaRCWA permittivity tensor with shape (Nw)
    wavelength_index:
        Position of the wavelength to select. Index zero selects the first
        wavelength.
    """

    flattened = value.detach().cpu().reshape(-1)

    return complex(flattened[wavelength_index].item())

@dataclass
class PreparedS4Model:
    """Initial S4 objects prepared from a MetaRCWA model.

    This first stage creates only the S4 simulation, lattice
    and Fourier settings. Materials, layer, geometry and illumination
    are added in later stages.

    Attributes
    ----------
    config:
        Common numerical settings translated into S4 compatible values.
    model_spec:
        MetRCWA's resolved model description. It contains evaluated
        materials, layers, lattice vectors, wavelengths and wavevectors.
    lattice_vectors:
        Two dimension lattice vectors converted into ordinary Python
        numbers for S4
    simulation:
        S4 simulation object configured with the lattice, requested 
        basis size and lattice truncation rule
    """

    config: S4Config
    model: Model
    model_spec: Any
    lattice_vectors: tuple[
        tuple[float,float],
        tuple[float,float]
    ]
    wavelength_index: int
    simulation: Any

    @classmethod
    def from_model(
        cls, 
        model: Model,
        config: Config,
        wavelength_index: int=0
        ) -> "PreparedS4Model":
        """Translate a MetaRCWA model into S4 simulation objects.

        This method
        
        - translates common numerical configuration to S4 compatible
        - resolves and rasterises the MetaRCWA model
        - creates the lattice vectors for S4
        - creates the simulation configured with the lattice, requested
        basis size and lattice truncation rule.
        """

        cfg = S4Config.from_config(config)

        # The original model contains higher-level description
        # such as shapes, layers and etc. Calling spec() evaluates
        # those descriptions such as evaluating permittivity at
        # requested wavelengths, nx, ny and etc.

        model_spec = model.spec(nx=cfg.nx,
                                ny=cfg.ny)

        # MetaRCWA stores lattice vectors as PyTorch tensors
        # S4 expects something of the form:
        # ((a1_x,a1_y), (a2_x,a2_y))
        # A nested tuple with two 2D lattice vectors
        lattice_vectors = (
            tensor_to_tuple(model_spec.a1),
            tensor_to_tuple(model_spec.a2)
        )

        simulation = S4.New(
            Lattice=lattice_vectors,
            NumBasis=cfg.requested_num_basis
        )

        simulation.SetOptions(LatticeTruncation= 
                            cfg.lattice_truncation)

        # S4 represents the semi-infinite incidence medium as the first
        # layer with zero thickness
        simulation.SetMaterial(
            Name="incidence",
            Epsilon=permittivity_at_wavelength(
                model_spec.incidence.eps,
                wavelength_index
            )
        )

        # Layer order follows the direction of incidence,
        # from top to bottom.
        simulation.AddLayer(
            Name="incidence",
            Thickness=0.0,
            Material="incidence"
        )

        # Add the finite layers in incidence to transmission order
        for layer_index, layer in enumerate(model_spec.layers):
            layer_name = f"layer_{layer_index}"

            if isinstance(layer, HomogeneousLayer):
                material_name = f"layer_{layer_index}_material"

                simulation.SetMaterial(
                    Name = material_name,
                    Epsilon=permittivity_at_wavelength(
                        layer.medium.eps,
                        wavelength_index
                    )
                )
            elif isinstance(layer, PatternedLayer):
                # S4 creates a patterned layer by first filling the
                # entire layer with its background material. Geometry
                # regions made from different material will be inserted 
                # in next implementation

                solid_material_name = f"layer_{layer_index}_solid"
                void_material_name = f"layer_{layer_index}_void"

                simulation.SetMaterial(
                    Name=solid_material_name,
                    Epsilon = permittivity_at_wavelength(
                        layer.medium_solid.eps,
                        wavelength_index
                    )
                )

                simulation.SetMaterial(
                    Name=void_material_name,
                    Epsilon = permittivity_at_wavelength(
                        layer.medium_void.eps,
                        wavelength_index
                    )
                )

                # Fill the complete patterned layer with the background material
                # The solid geometry will be inserted after
                material_name = void_material_name

            else:
                raise TypeError(
                    "Unsupported MetaRCWA layer type:"
                    f"{type(layer).__name__}"
                )
            
            # Add the layers to the stack one by one
            simulation.AddLayer(
                Name=layer_name,
                Thickness=tensor_to_float(layer.thickness),
                Material=material_name
            )
        
        # Set the final semi-infinite transmission medium as the final
        # zero thickness layer at the end of the stack
        simulation.SetMaterial(
            Name="transmission",
            Epsilon=permittivity_at_wavelength(
                model_spec.transmission.eps,
                wavelength_index
            )
        )

        simulation.AddLayer(
            Name="transmission",
            Thickness=0.0,
            Material="transmission"
        )

        return cls(
            config=cfg,
            model_spec=model_spec,
            lattice_vectors=lattice_vectors,
            simulation=simulation,
            wavelength_index=wavelength_index,
            model=model
        )