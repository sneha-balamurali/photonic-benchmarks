"""
Load YAML benchmark data and convert it into MetaRCWA objects.

The YAML file contains ordinary values such as numbers, names and lists.
This module converts those values into a MetaRCWA physical model and numerical 
Config that can later be passed to the solver backend.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from metarcwa import (Config as MetaRCWAConfig,
                      IsotropicMedium,
                      Lattice,
                      Medium,
                      Layer,
                      Stack,
                      Source,
                      Stack)

@dataclass
class S4Settings:
    """
    Numerical settings used only by the S4 backend

    The initial version contains only the basis size. Additional supported
    S4 SetOptions values can be added after checking the existing S4 code.
    """

    basis_size: int

    def to_dict(self) -> dic:
        """Converts the S4 setting into an ordinary dictionary."""
        return{"basis_size:" self.basis_size}

    @classmethod
    def from_dict(cls, data:dict) -> "S4Settings":
        """
        Construct S4 settings from a dictionary loaded from YAML
        """
        return cls(basis_size=int(data["basis_size"]))

@dataclass
class FMMaxSettings:
    """
    Numerical settings used only by the FMMax backend.
    """
    approximate_num_terms: int
    
    def to_dict(self) -> dic: 
        """Convert the FMMax settings into an ordinary dictionary
        """
        return{"approximate_num_terms": self.approximate_num_terms}

    @classmethod
    def from_dict(cls, data:dict) -> "FMMaxSettings":
        """
        Construct FMMax from a dictionary loaded by YAML
        """
        return cls(approximate_num_terms=int(data["approximate_num_terms"]))

@dataclass
class BenchmarkConfig:
    """
    Convert plain YAML data into typed Python objects in order to have the 
    complete input required to run one benchmark through to the solver backends.

    Attributes
    ----------
    name: 
        Human-readable name of the benchmark
    model: 
        Solver-independent physical structure constructed with MetaRCWA
    numerics:
        Numerical grid, Fourier-order and truncation settings
    s4_settings:
        Settings that apply only to S4 backened
    fmmax_settings:
        Settings that apply only to FMMax backened
    """
    name: str
    model: Model
    numerics: MetaRCWAConfig
    s4_settings: S4Settings
    fmmax_settings: FMMaxSettings

def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """
    Load one YAML file and create a complete benchmark configuration.

    Parameters:
    -----------
    path: str | Path
        Location of the YAML configuration file.
    
    Returns:
    --------
    BenchmarkConfig:
        The physical model and numerical settings ready for the backend.
    """

    # yaml.safe_load() converts the YAML text into a Python dictionary
    with open(path) as file:
        data = yaml.safe_load(file)

    # Build the two main parts of the benchmarks
    model = build_model(data)
    numerics = build_numerics(data["numerics"])
 
    return BenchmarkConfig(
        name=data["name"],
        model=model,
        numerics=numerics,
        s4_settings=S4Settings.from_dict(data["s4"]),
        fmmax_settings=FMMaxSettings.from_dict(data["fmmax"]),
    )

def build_model(data: dict[str,Any]) -> Model:
    """
    Construct a MetaRCWA Model from the physical YAML sections.

    This function will create the materials, lattice, layers and stack
    and source.

    At this point, it shouldn't contain any solver specific operations.
    """

    # ToDo: Implement the YAML-to-MetaRCWA conversion here
        
    pass

def build_numerics(data: dict[str,Any]) -> MetaRCWAConfig:
    """Construct MetaRCWA numerical settings from the YAML dictionary."""
    pass
    