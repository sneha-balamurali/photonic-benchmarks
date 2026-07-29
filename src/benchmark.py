
from metarcwa import Model
from src.config import Config

class Benchmark:
    """ Main class to run benchmarks"""
    def __init__(self, model: Model, config: Config):
        self.model = model
        self.config = config
        
    def spectra_comparison(self, solvers: list):
        """ Compute spectra with different solvers and compare them"""
        pass
    
    def convergence_test(self, N_max: int, solvers: list):
        """ Run convergence tests with different solvers"""
        pass