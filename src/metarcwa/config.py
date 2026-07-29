from dataclasses import dataclass

from src.config import Config
from metarcwa.config import Config as metarcwa_config

@dataclass
class MetaRCWAConfig(metarcwa_config):
    """ Subclass of the metarcwa.config class for config adapter"""
    @classmethod
    def from_config(config: Config):
        """ Prepares metarcwa config from the standard config. """
        pass