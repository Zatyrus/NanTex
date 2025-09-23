## Dependencies
import torch
import numpy as np

from torch.utils.data import Dataset
from typing import Tuple, Dict, List, NoReturn, Any

## Custom Dependencies
from ..Util.bit_generator_utils import initialize_generator, seed_generator


class FileGrabber(Dataset):
    # attributes
    _files: List[str]
    _patchsize: Tuple[int, int]
    _in_channels: int
    _out_channels: int

    _dtype_overlay_out: np.dtype
    _dtype_masks_out: np.dtype

    _gen_type: str
    _gen_seed: int
    _gen: np.random.Generator

    _num_shuffle: int

    "Load, Augment and distribute batches for training & valitation."

    def __init__(
        self,
        files: List[str],
        patchsize: Tuple[int, int],
        in_channels: int = 1,
        out_channels: int = 3,
        dtype_overlay_out: np.dtype = np.float32,
        dtype_masks_out: np.dtype = np.float32,
        gen_type: str = "DXSM",
        gen_seed: int = None,
        num_shuffle: int = 7,
    ) -> None:
        """Data generator object used in training and validation to load, augment and distribute raw and validation data in batch.

        Args:
            files (list): List of overlayed data including a single-color, multi-structure image in addition to the single ground truth images.
            dim (tuple, optional): Patch Dimension (row, column). Defaults to (256,256).
            out_channels (int, optional): Number of output channels. Defaults to 3.
            batchsize (int, optional): Number of patches per batch. Defaults to 32.
            dtype_overlay_out (np.dtype, optional): Data type of the overlay output data. Defaults to np.float32.
            dtype_masks_out (np.dtype, optional): Data type of the masks. Defaults to np.float32.
            gen_type (str, optional): Type of random number generator. Defaults to 'DXSM'.
            gen_seed (int, optional): Seed for the random number generator. Defaults to None.
            is_val (bool, optional): Flag if the object is used for generating validation data. Defaults to False.
        """

        "Initialization"

        ## content
        self._files = files

        # format info
        self._patchsize = patchsize
        self._in_channels = in_channels
        self._out_channels = out_channels

        # metainformation
        self._dtype_masks_out = dtype_masks_out
        self._dtype_overlay_out = dtype_overlay_out

        # randomization
        self._gen_type = gen_type
        self._gen_seed = gen_seed
        self._num_shuffle = num_shuffle

        # behavioral flags
        ...

        # post init routines
        self.__post_init__()

    def __post_init__(self) -> NoReturn:
        """Post-initialization routine to set up the dataset.

        Returns:
            NoReturn: This function does not return a value.
        """
        self.__initialize_generator__()
        self.__shuffle_paths__()

    def __initialize_generator__(self) -> NoReturn:
        """Initialize the random number generator based on the specified type and seed.

        Returns:
            NoReturn: This function does not return a value.
        """

        if self._gen_seed == None:
            self._gen = initialize_generator(self._gen_type)
        else:
            self._gen = seed_generator(self._gen_type, self._gen_seed)

    def __shuffle_paths__(self) -> NoReturn:
        """Shuffle the file paths in the dataset. This method shuffles the file paths a specified number (self._num_shuffle) of times to ensure randomness.

        Returns:
            NoReturn: This function does not return a value.
        """
        while self._num_shuffle:
            # shuffle filepaths
            self._gen.shuffle(self._files)

            # reduce counter
            self._num_shuffle -= 1

    def __len__(self) -> int:
        """Get the total number of files in the dataset.

        Returns:
            int: Total number of files.
        """
        return len(self._files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch a single data sample and its corresponding mask.

        Args:
            index (int): Index of the data sample to fetch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The data sample and its corresponding mask.
        """

        # initialize stack to reserve memory
        X, y = self.__initialize_stack__()

        # grab and load data
        X, y = self.__fetch_data__(index)

        return X, y

    def __initialize_stack__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize the data and mask tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The initialized data and mask tensors.
        """
        X = torch.from_numpy(
            np.empty(
                (1, self._in_channels, *self._patchsize),
                dtype=self._dtype_overlay_out,
            )
        )
        y = torch.from_numpy(
            np.empty(
                (1, self._out_channels, *self._patchsize),
                dtype=self._dtype_masks_out,
            )
        )
        return X, y

    def __fetch_data__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch a single data sample and its corresponding mask.

        Args:
            index (int): Index of the data sample to fetch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The data sample and its corresponding mask.
        """
        # grab data
        tmp = np.load(self._files[index])

        # Extract Img and Masks
        X = tmp[self._out_channels :]
        y = tmp[: self._out_channels]

        # ensure data type
        X = torch.from_numpy(X.astype(self._dtype_overlay_out))
        y = torch.from_numpy(y.astype(self._dtype_masks_out))

        return X, y

    # %% Dunder Methods
    def get_dict(self) -> Dict[Any, Any]:
        """Get the internal state of the object as a dictionary.

        Returns:
            Dict[Any, Any]: Internal state of the object.
        """
        return self.__dict__

    # %% Helper
