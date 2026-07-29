
import jax
import torch
from typing import Tuple

from src.fmmax.config import FMMaxConfig
from metarcwa import Model, Config


def run_fmmax(model: Model, config: Config) -> Tuple[torch.Tensor, torch.Tensor,
                                                        torch.Tensor, torch.Tensor]:
    """ Runs fmmax computation. Need to declare 32/64 bits and device before running. """
    # Translate the config
    cfg = FMMaxConfig.from_config(config)
    ...
    
    # Replace for real computation
    Rs = torch.zeros()
    Rp = torch.zeros()
    Ts = torch.zeros()
    Tp = torch.zeros()
    return Rs, Rp, Ts, Tp