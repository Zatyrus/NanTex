## Dependencies
import torch
import numpy as np

from torch.utils.data import Dataset
from typing import Tuple, Dict, List, NoReturn, Any

## Custom Dependencies
from nantex.core import BitGenCore as BGC


class Euthenia(Dataset):
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
    _file_count_multiplier: int

    # %% properties
    @property
    def files(self) -> List[str]:
        return self._files

    @files.setter
    def files(self, value: List[str]) -> NoReturn:
        self._files = value
        self.__post_init__()

    @property
    def patchsize(self) -> Tuple[int, int]:
        return self._patchsize

    @patchsize.setter
    def patchsize(self, value: Tuple[int, int]) -> NoReturn:
        self._patchsize = value

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @in_channels.setter
    def in_channels(self, value: int) -> NoReturn:
        self._in_channels = value

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @out_channels.setter
    def out_channels(self, value: int) -> NoReturn:
        self._out_channels = value

    @property
    def dtype_overlay_out(self) -> np.dtype:
        return self._dtype_overlay_out

    @dtype_overlay_out.setter
    def dtype_overlay_out(self, value: np.dtype) -> NoReturn:
        self._dtype_overlay_out = value

    @property
    def dtype_masks_out(self) -> np.dtype:
        return self._dtype_masks_out

    @dtype_masks_out.setter
    def dtype_masks_out(self, value: np.dtype) -> NoReturn:
        self._dtype_masks_out = value

    @property
    def gen_type(self) -> str:
        return self._gen_type

    @gen_type.setter
    def gen_type(self, value: str) -> NoReturn:
        self._gen_type = value
        self.__initialize_generator__()

    @property
    def gen_seed(self) -> int:
        return self._gen_seed

    @gen_seed.setter
    def gen_seed(self, value: int) -> NoReturn:
        self._gen_seed = value
        self.__initialize_generator__()

    @property
    def num_shuffle(self) -> int:
        return self._num_shuffle

    @num_shuffle.setter
    def num_shuffle(self, value: int) -> NoReturn:
        self._num_shuffle = value
        self.__shuffle_paths__()

    @property
    def file_count_multiplier(self) -> int:
        return self._file_count_multiplier

    @file_count_multiplier.setter
    def file_count_multiplier(self, value: int) -> NoReturn:
        self._file_count_multiplier = value
        self.__apply_file_count_multiplier__()
        self.__shuffle_paths__()

    "Load, Augment and distribute batches for training & validation."

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
        file_count_multiplier: int = 1,
    ) -> "Euthenia":
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
            num_shuffle (int, optional): Number of times to shuffle the file paths. Defaults to 7.
            file_count_multiplier (int, optional): Multiplier to increase the effective number of files. Defaults to 1.
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
        self._file_count_multiplier = file_count_multiplier

        # post init routines
        self.__post_init__()

    def __post_init__(self) -> NoReturn:
        """Post-initialization routine to set up the dataset.

        Returns:
            NoReturn: This function does not return a value.
        """
        self.__initialize_generator__()
        self.__apply_file_count_multiplier__()
        self.__shuffle_paths__()

    def __initialize_generator__(self) -> NoReturn:
        """Initialize the random number generator based on the specified type and seed.

        Returns:
            NoReturn: This function does not return a value.
        """

        if self._gen_seed is None:
            self._gen = BGC.initialize_generator(self._gen_type)
        else:
            self._gen = BGC.seed_generator(self._gen_type, self._gen_seed)

    def __apply_file_count_multiplier__(self) -> NoReturn:
        """Apply the file count multiplier to increase the effective number of files.

        Returns:
            NoReturn: This function does not return a value.
        """
        self._files = self._files * self._file_count_multiplier

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
