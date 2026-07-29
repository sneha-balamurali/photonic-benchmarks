from dataclasses import dataclass

from src.config import Config

@dataclass
class FMMaxConfig:
    pass
    
    @classmethod
    def from_config(config: Config):
        """ Prepares FMMax config from the standard config. """
        pass