## Dependencies
import numpy as np

# typing
from typing import List, Dict


class BitGenCore:
    gen_types: Dict[str, np.random.Generator] = {
        "default": np.random.MT19937,
        "PCG64": np.random.PCG64,
        "DXSM": np.random.PCG64DXSM,
    }

    @staticmethod
    def initialize_generator(gen_type: str) -> None:
        return np.random.Generator(BitGenCore.gen_types[gen_type].__call__(seed=None))

    @staticmethod
    def seed_generator(gen_type: str, seed: int) -> None:
        return np.random.Generator(BitGenCore.gen_types[gen_type].__call__(seed=seed))

    @staticmethod
    def spawn_generator(generator: np.random.Generator) -> List[np.random.Generator]:
        return generator.spawn(1)[0]

    @staticmethod
    def default_rng() -> np.random.Generator:
        return np.random.default_rng()

    @staticmethod
    def seed_rng(seed: int) -> np.random.Generator:
        return np.random.default_rng(seed=seed)
