from gaussed.gp.backends.base import Backend
from gaussed.linops import LinearOperator

from typing import Dict, Callable


_BACKENDS: Dict[str, Callable[..., Backend]] = {}

def register_backend(name: str):
    def deco(factory: Callable[..., Backend]):
        _BACKENDS[name] = factory
        return factory
    return deco

def create_backend(name: str, **kwargs) -> Backend:
    try:
        return _BACKENDS[name](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown backend '{name}'. Available: {list(_BACKENDS)}")
