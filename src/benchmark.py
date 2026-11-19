from metarcwa import Model

from src.config import Config
from src.fmmax.simulation import run_fmmax
from src.s4.simulation import run_s4
from src.metarcwa.simulation import run_metarcwa

def run_backends(
        model: Model,
        config: Config
) -> dict: 
    """
    Run every supported backend with the same model 
    and configuration
    """

    return {
        "fmmax": run_fmmax(model,config),
        "s4": run_s4(model,config),
        "metarcwa": run_metarcwa(model,config)
    }

