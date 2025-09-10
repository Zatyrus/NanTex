## Dependencies
import torch
import numpy as np

from torch.utils.data import Dataset
from typing import Tuple, Dict, List, NoReturn

## Custom Dependencies
from ..Util.bit_generator_utils import initialize_generator, seed_generator


class FileGrabber(Dataset):
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
        num_shuffle:int = 7
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
        self.__initialize_generator__()
        self.__shuffle_paths__()

    def __initialize_generator__(self)->NoReturn:
        ## Initialize Bit Generator
        self._gen: np.random.Generator
        if self._gen_seed == None:
            self._gen = initialize_generator(self._gen_type)
        else:
            self._gen = seed_generator(self._gen_type, self._gen_seed)

    def __shuffle_paths__(self)->NoReturn:
        while self._num_shuffle:
            # shuffle filepaths
            self._gen.shuffle(self._files)
            
            # reduce counter
            self._num_shuffle -= 1

    def __len__(self) -> int:
        """Denotes the number of files per batch

        Returns:
            int: Number of files in batch
        """
        return len(self._files)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate one batch of data

        Args:
            index (int): index of file in dataset to load

        Returns:
            np.ndarray: X - samples, y - ground truth
        """

        # initialize stack to reserve memory
        X, y = self.__initialize_stack__()

        # grab and load data
        X, y = self.__fetch_data__(index)

        return X, y

    def __initialize_stack__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # setup the batch
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
        """Generates data containing # batch_size augmented samples and likewise augmented ground truth.

        Returns:
            np.ndarray: X - augmentd sample stack,
            y - augmented ground truth stack
        """

        # grab data
        tmp = np.load(self._files[index])

        # Extract Img and Masks
        X = tmp[self._out_channels :]
        y = tmp[: self._out_channels]
        
        # ensure data type
        X = torch.from_numpy(X.astype(self._dtype_overlay_out))
        y = torch.from_numpy(y.astype(self._dtype_masks_out))

        # if self._in_channels == 1:
        #     X = X[torch.newaxis, ...]

        # if self._out_channels == 1:
        #     y = y[torch.newaxis, ...]

        return X, y

    # %% Dunder Dethods
    def get_dict(self) -> Dict:
        return self.__dict__

    # %% Helper