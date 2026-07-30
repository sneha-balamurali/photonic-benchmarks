from dataclasses import dataclass

import torch

from src.config import Config
from metarcwa import Config as metarcwa_config

_DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64
    }

@dataclass
class MetaRCWAConfig(metarcwa_config):
    """ Subclass of the metarcwa.config class for config adapter"""
    @classmethod
    def from_config(cls, config: Config) -> "MetaRCWAConfig":
        """ Prepares metarcwa config from the standard config. """

        return cls(
            dtype=DTYPE_MAP[config.dtype],
            device=config.device,
            nx=config.nx,
            ny=config.ny,
            m=config.m,
            n=config.n,
            truncation=config.truncation,

            # Temporary baseline for the first working route
            # Can expose MetaRCWA factorisation and etc later
            factorization=None,
        )
        