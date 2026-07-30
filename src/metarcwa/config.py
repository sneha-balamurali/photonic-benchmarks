from dataclasses import dataclass

import torch

from src.config import Config
from metarcwa.config import Config as metarcwa_config

@dataclass
class MetaRCWAConfig(metarcwa_config):
    """ Subclass of the metarcwa.config class for config adapter"""
    @classmethod
    def from_config(config: Config):
        """ Prepares metarcwa config from the standard config. """

        return cls(
            dtype=config.dtype
            device=config.device
            nx=config.nx
            ny=config.ny
            m=config.m
            n=config.n
            truncation=config.truncation

        # Temporary baseline for the first working route
        # Can expose MetaRCWA factorisation and etc later
        )
        