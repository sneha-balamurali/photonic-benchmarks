from dataclasses import dataclass

from src.config import Config

@dataclass
class S4Config:
    pass
    
    @classmethod
    def from_config(config: Config):
        """ Prepares S4 config from the standard config. """
        pass