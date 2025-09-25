## Dependencies
import numpy as np
from typing import List, Dict

gen_types: Dict[str, np.random.Generator] = {
    "default": np.random.MT19937,
    "PCG64": np.random.PCG64,
    "DXSM": np.random.PCG64DXSM,
}


def initialize_generator(gen_type: str) -> None:
    return np.random.Generator(gen_types[gen_type].__call__(seed=None))


def seed_generator(gen_type: str, seed: int) -> None:
    return np.random.Generator(gen_types[gen_type].__call__(seed=seed))


def spawn_generator(generator: np.random.Generator) -> List[np.random.Generator]:
    return generator.spawn(1)[0]


def default_rng() -> np.random.Generator:
    return np.random.default_rng()


def seed_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed=seed)
