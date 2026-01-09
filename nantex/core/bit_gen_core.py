## Dependencies
import numpy as np

# typing
from typing import Dict


class BitGenCore:
    gen_types: Dict[str, np.random.Generator] = {
        "default": np.random.MT19937,
        "PCG64": np.random.PCG64,
        "DXSM": np.random.PCG64DXSM,
    }

    @staticmethod
    def initialize_generator(gen_type: str) -> np.random.Generator:
        """Instantiate numpy BitGenerator based on an input string without seed.

        Args:
            gen_type (str): BitGenerator type. See numpy documentation.

        Returns:
            np.random.Generator: Returns the instantiated BitGenerator.
        """
        return np.random.Generator(BitGenCore.gen_types[gen_type].__call__(seed=None))

    @staticmethod
    def seed_generator(gen_type: str, seed: int) -> np.random.Generator:
        """Instantiate numpy BitGenerator with seed.

        Args:
            gen_type (str): BitGenerator type. See numpy documentation.
            seed (int): RNG seed.

        Returns:
            np.random.Generator: Returns the instantiated BitGenerator.
        """
        return np.random.Generator(BitGenCore.gen_types[gen_type].__call__(seed=seed))

    @staticmethod
    def spawn_generator(generator: np.random.Generator) -> np.random.Generator:
        """Spawn new child generator from numpy Bit Generator.

        Args:
            generator (np.random.Generator): Child BitGenerator

        Returns:
            np.random.Generator: Child BitGenerator
        """
        return generator.spawn(1)[0]

    @staticmethod
    def default_rng() -> np.random.Generator:
        """Instantiate default BitGenerator (PCG64).

        Returns:
            np.random.Generator: PCG64 BitGenerator.
        """
        return np.random.default_rng()

    @staticmethod
    def seed_rng(seed: int) -> np.random.Generator:
        """Instantiate default BitGenerator (PCG64) with seed.

        Args:
            seed (int): RNG seed.

        Returns:
            np.random.Generator: PCG64 BitGenerator.
        """
        return np.random.default_rng(seed=seed)
