from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    """
    Solver-independent numerical settings for one simulation.
    """
    
    dtype: str
    device: str
    nx: int
    ny: int
    m: int
    n: int
    truncation: str

    # Check for errors

    def __post_init__(self) -> None:
        """Check the supplied values for simple mistakes
        """
    
        if self.nx <=0 or self.ny <=0:
            raise ValueError("nx and ny must be positive")

        if self.m < 0 or self.n < 0:
            raise ValueError("m and n cannot be negative")

        if self.dtype not in ("float32","float64"):
            raise ValueError("dtype must be 'float32' or 'float64'")

        if self.truncation not in ("circular","rectangular", "parallelogramic"):
            raise ValueError("truncation must be 'circular' or 'rectangular' or 'parallelogramic")


    def to_dict(self) -> dict:
        """
        Convert the configuration into a dictionary.
        """

        return{
            "dtype": self.dtype,
            "device": self.device,
            "nx": self.nx,
            "ny": self.ny,
            "m": self.m,
            "n": self.n,
            "truncation": self.truncation,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """
        Create a configuration from YAML data.
        """
        return cls(**d)