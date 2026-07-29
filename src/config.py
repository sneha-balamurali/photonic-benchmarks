from dataclasses import dataclass

@dataclass
class Config:
    pass

    def to_dict(self) -> dict:
        pass
    
    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        pass