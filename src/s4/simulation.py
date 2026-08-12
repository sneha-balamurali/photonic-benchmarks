from dataclasses import dataclass
from typing import Any

import S4
import torch
from metarcwa import Model

from src.config import Config
from src.s4.config import S4Config
from src.s4.geometry import add_shape, metashape_from_layer

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
    wavelength_index:
        Position of the selected wavelength in the MetaRCWA wavelength sweep.
        Index zero selects the first wavelength.
    wavelength:
        Selected wavelength as an Python float. Uses the same length unit as
        the model geometry.
    """

    config: S4Config
    model: Model
    model_spec: Any
    lattice_vectors: tuple[
        tuple[float,float],
        tuple[float,float]
    ]
    wavelength_index: int
    wavelength: float
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

        # MetaRCWA stores all the requested wavelengths in a PyTorch tensor
        # e.g. tensor([500,600])
        # One S4 simulation object uses one frequency at at time, so select
        # the wavelength indicated by wavelength_index.

        wavelengths = (model_spec.wavelength
                       .detach()
                       .cpu()
                       .reshape(-1)
                       )

        number_of_wavelengths = wavelengths.numel()
        
        # Check the index and give a clear explanation if 
        # there is an indexing error
        if not 0<= wavelength_index < number_of_wavelengths:
            raise IndexError(
                "wavelength_index is outside the model's wavelength array. "
                f"Received {wavelength_index}, but model contains "
                f"{number_of_wavelengths} wavelength values."
            )

        # item() extracts one number from a one value PyTorch tensor
        # float() converts that number into a float
        # which is what the S4 Python interface expects.
        wavelength = float(
            wavelengths[wavelength_index].item()
        )

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

        # S4 expects frequency = 1 / wavelength, rather than 
        # wavelength itself or angular frequency.
        # All lengths must use one consistent unit. In this model,
        # both geometry and wavelength use nanometres, so the
        # numerical frequency is expressed in inverse nanometres.
        simulation.SetFrequency(
            1.0 / wavelength
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
        # MetaRCWA model_spec defines layer 0 from the first finite layer
        # The semi-infinite incidence and transmission layers are defined
        # seperately 
        for layer_index, layer in enumerate(model_spec.layers):
            layer_name = f"layer_{layer_index}"

            if isinstance(layer, HomogeneousLayer):
                # A homogeneous layer contains only one material 
                # and doesn't require an analytical geometry region
                material_name = f"layer_{layer_index}_material"

                simulation.SetMaterial(
                    Name = material_name,
                    Epsilon=permittivity_at_wavelength(
                        layer.medium.eps,
                        wavelength_index
                    )
                )

                simulation.AddLayer(
                    Name=layer_name,
                    Thickness=tensor_to_float(layer.thickness),
                    Material=material_name
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

                # Begin by filling the layer with the 
                # void/background material
                simulation.AddLayer(
                    Name=layer_name,
                    Thickness=tensor_to_float(layer.thickness),
                    Material=void_material_name
                )

                # The resolved layer contains a raster mask, 
                # while the unresolved layer contains the original
                # MetaShapes geometry.
                original_layer = model.stack.layers[layer_index]

                shape = metashape_from_layer(original_layer)

                # Insert the solid analytical geometry into the background
                # layer. The mapping geometry.py selects add_rectangle()
                # for the current Rectangle. 
                add_shape(
                    simulation,
                    layer_name=layer_name,
                    material_name=solid_material_name,
                    shape=shape,
                    lattice_vectors=lattice_vectors
                )

            else:
                raise TypeError(
                    "Unsupported MetaRCWA layer type:"
                    f"{type(layer).__name__}"
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
            model=model,
            wavelength=wavelength
        )

@dataclass
class S4DiffractionResult:
    """Diffraction order powers returned by S4
    
    S4 is run twice at every wavelength and angle:
    - once with an s-polarised incident wave
    - once with a p-polarised incident wave
    
    Each output tensor stores the power in every retained
    diffraction order. The last tensor axis corresponds to
    these orders.

    Attributes
    ----------
    orders:
        Retained S4 diffraction-order pairs (m,n). If 
        orders[3] == (1,-1), then index 3 on every power tensor
        contains the power in diffraction order (1,-1).
    reflection_s_incident:
        Reflected power per order for an s-polarised incident 
        wave. Shape (Nw, Ntheta, Nphi, Norders)
    reflection_p_incident:
        Reflected power per order for a p-polarised incident 
        wave. Shape (Nw, Ntheta, Nphi, Norders)
    transmission_s_incident:
        Transmitted power per order for a s-polarised incident
        wave. Shape (Nw, Ntheta, Nphi, Norders)
    transmission_p_incident:
        Transmitted power per order for a p-polarised incident
        wave. Shape (Nw, Ntheta, Nphi, Norders)

    Notes
    -----
    S4's GetPowerFluxOrder() reports total power in each diffraction
    order. It doesn't seperate the outgoing order into s and p 
    polarised components.
    """

    orders: tuple[tuple[int,int],...]

    reflection_s_incident: torch.Tensor
    reflection_p_incident: torch.Tensor
    transmission_s_incident: torch.Tensor
    transmission_p_incident: torch.Tensor

    def powers_for_order(self,
                         order: tuple[int,int]
                    ) -> tuple[torch.Tensor,
                               torch.Tensor,
                               torch.Tensor,
                               torch.Tensor]:
        """Return s and p incident powers for one diffraction order.

        Paramters
        ---------
        order: 
            Diffraction-order pair (m,n) such as (0,0),
            (1,0), (1,-1)
        
        Returns
        -------
        Rs, Rp, Ts, Tp:
            Power tensors with shape (Nw, Ntheta, Nphi).
            Where s and p in the subscript indicate the polarisation
            of the incident wave. S4's per order flux doesn't seperate
            the output wave into co and cross polarised parts. 
        """

        # orders is a tuple such as:
        # ((0,0), (0,-1), (-1,0),...)
        # Find the integer position of the requested pair
        try:
            order_index = self.orders.index(order)
        except ValueError as error:
            raise ValueError(
                f"Diffraction order {order!r} was not retained by S4. "
                f"Retained orders: {self.orders!r}"
            ) from error

        # Each complete power tensor has the shape:
        # (Nw, Ntheta, Nphi, Norders)
        # order_index selects one diffraction order from the final axis
        Rs=self.reflection_s_incident[...,order_index]
        Rp=self.reflection_p_incident[...,order_index]
        Ts=self.transmission_s_incident[...,order_index]
        Tp=self.transmission_p_incident[...,order_index]

        return Rs, Rp, Ts, Tp
