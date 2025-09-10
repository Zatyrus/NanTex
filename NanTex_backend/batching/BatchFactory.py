## Dependencies
import os
import glob
import json
import numpy as np
from pprint import pprint

from torch.utils.data import DataLoader

from typing import Tuple, Union, Dict, List

## Custom Dependencies
from ..Util.pyDialogue import pyDialogue as pD
from .FileGrabber import FileGrabber


# %% Convenience Class
class BatchFactory:
    config: Dict[str, Union[str, int, float, bool]]
    data_path_container: Dict[str, List[str]]
    datatype: str
    DEBUG: bool

    def __init__(
        self, config: Dict = None, datatype: str = "npy", DEBUG: bool = False
    ) -> None:
        self.config = config
        self.DEBUG = DEBUG
        self.datatype = datatype
        self.data_path_container = {"raw_source": None, "val_source": None}

        self.__post_init__()

    def __post_init__(self) -> None:
        self.__config_startup__()
        self.__check_datatype__()
        self.__check_config__()

        # Check if data sources are available
        if not self.__check_sources__():
            if self.DEBUG:
                print("Data sources are not available.")
            self.__call_sources__()

    @classmethod
    def from_config(
        cls, config_file_path: str = None, datatype: str = "npy", DEBUG: str = False
    ) -> "BatchFactory":
        # Fetch the configuration file path
        if config_file_path == None:
            config_file_path = pD.askFILE(
                "Please provide the path to the configuration file."
            )

        # Load the configuration file
        with open(config_file_path, "r") as f:
            config = json.load(f)
        return cls(config, datatype, DEBUG)

    # %% Convenience Functions
    @staticmethod
    def generate_biolerplate_config_file(outpath: str = None) -> None:
        """
        Generate a configuration file for the data generator object.
        This file can be used to store the configuration of the data generator object for later use.

        Returns:
            dict: Configuration file
        """

        if outpath == None:
            outpath = pD.askDIR("Please provide the path to the output directory.")

        # set the configuration dictionary
        config = {
            "raw_source": None,  # Required
            "val_source": None,  # Required
            "train_batchsize": 32,  # Number of patches per batch. <- play with it if you want, but PLEASE keep it a power of 2.
            "val_batchsize": 8,  # Might want to change that if you play with the validation scheme. Leave 1 if you have an image (not patch) based validation.
            "train_shuffle_per_draw": True,  # LEAVE TRUE, unless you know what you are doing
            "val_shuffle_per_draw": True,  # LEAVE TRUE, unless you know what you are doing
            "num_shuffle_train": 7,  # Number of times the training data is shuffled on setup
            "num_shuffle_val": 7,  # Number of times the validation data is shuffled on setup
            "patchsize": (
                256,
                256,
            ),  # Patch Dimension for training (row, column) <- play with it, but PLEASE keep it a power of 2.
            "num_train_workers": 4,  # Number of workers <- run the detect_workers() function to get a recommendation. I hope you got more than 4.
            "num_val_workers": 2,  # Number of workers for validation <- run the detect_workers() function to get a recommendation. This can be lower than the training workers.
            "prefetch": 8,  # Number of batches to prefetch <- based on your computational resources.
            "in_channels": 1,  # Number of input channels <- 1 for grayscale, 3 for RGB | For the sake of NanTex, we use 1. Leave it.
            "out_channels": 3,  # Number of output channels <- Number of structures to segment.
            "dtype_overlay_out": "float32",  # Data type of the output data <- Leave it.
            "dtype_masks_out": "float32",  # Data type of the masks <- Leave it.
            "gen_type": "DXSM",  # Type of random number generator <- Leave it, if you don't know what you are doing.
            "gen_seed": None,  # Seed for the random number generator <- Use for reproducibility.
            "pin_memory": True,  # Flag to pin memory <- Leave it.
            "persistant_workers": True,  # Flag to keep workers alive <- Leave it.
        }

        # Dump the configuration file
        with open(f"{outpath}/BatchFactory_config.json", "w") as f:
            json.dump(config, f, indent=4, sort_keys=False)

    # %% Factory Functions
    def build(self) -> Tuple[DataLoader, DataLoader]:
        # run checkups
        if self.DEBUG:
            print("Running checks.")
        try:
            assert self.__check__()
        except Exception as e:
            print(e)
            return None

        # fetch data
        if self.DEBUG:
            print("Fetching data.")
        self.__load_sources__()

        if self.DEBUG:
            print("Building Data Loader.")
        return self.__build_BatchFactory__(**self.config | self.data_path_container)

    # %% Data Loader Backend
    def __build_BatchFactory__(
        self,
        raw_source: str,
        val_source: str = None,
        patchsize=(256, 256),
        in_channels=1,
        out_channels=3,
        train_batchsize: int = 32,
        val_batchsize: int = 8,
        train_shuffle_per_draw: bool = True,
        val_shuffle_per_draw: bool = True,
        num_shuffle_train: int = 7,
        num_shuffle_val: int = 7,
        num_train_workers: int = 6,
        num_val_workers: int = 2,
        prefetch: int = 8,
        pin_memory: bool = True,
        persistant_workers: bool = True,
        dtype_overlay_out: np.dtype = np.float32,
        dtype_masks_out: np.dtype = np.float32,
        gen_type: str = "DXSM",
        gen_seed: int = None,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Setup DataLoad objects and feed them with data.
        These objects are used to continuously generate batches for both training and validation.
        Multi processing with persistent workers is highly recommended.

        Args:
            raw_source (str): Path to raw data.
            val_source (str, optional): Path to validation data if applicable. Defaults to None.
            val_split (float, optional): Percentage of raw data to use as a substitution for validation. This is NOT recommended, but better than no validation when heavy augmentation is applied to the data. Defaults to 0.1.
            patchsize (tuple, optional): Set raw patch dimensions in px. (row,column). Defaults to (256,256).
            in_channels (int, optional): Number of input channels. Each channel is used for one feature only. (e.g. gray scale image -> 1 in-channel). Defaults to 1.
            out_channels (int, optional): Number of output channels. Is equal to the number of structures to separate. Defaults to 3.
            train_batchsize (int, optional): Number of patches per batch. Defaults to 32.
            val_batchsize (int, optional): Set the number of batches used in validation. Defaults to 1.
            train_shuffle_on_draw (bool, optional): Flag to shuffle datasets between batches. Recommended. Defaults to True.
            val_shuffle_on_draw (bool, optional): Flag to shuffle datasets between batches. Recommended. Defaults to True.
            num_shuffle_train (int, optional): Number of times the training data is shuffled on setup. Defaults to 7.
            num_shuffle_val (int, optional): Number of times the validation data is shuffled on setup. Defaults to 7.
            num_train_workers (int, optional): Number of workers for training. Defaults to 6.
            num_val_workers (int, optional): Number of workers for validation. Defaults to 2.
            prefetch (int, optional): Set how many batches should be queued per worker. Defaults to 8.
            pin_memory (bool, optional): Flag to pin memory. Defaults to True.
            persistant_workers (bool, optional): Flag to keep workers alive. Defaults to True.
            dtype_overlay_out (np.dtype, optional): Data type of the output data. Defaults to np.float32.
            dtype_masks_out (np.dtype, optional): Data type of the masks. Defaults to np.uint8.
            gen_type (str, optional): Type of random number generator. Defaults to 'DXSM'.
            gen_seed (int, optional): Seed for the random number generator. Defaults to None.
        Returns:
            DataLoader: _description_
        """
        if self.DEBUG:
            print("Building Data Loader.")

        if val_source == None:
            raise ValueError("Validation source must be provided.")

        ## Retrun FileGrabber objects for train and test data
        train_dataset = FileGrabber(
            files=raw_source,
            patchsize=patchsize,
            in_channels=in_channels,
            out_channels=out_channels,
            dtype_overlay_out=dtype_overlay_out,
            dtype_masks_out=dtype_masks_out,
            gen_type=gen_type,
            gen_seed=gen_seed,
            num_shuffle=num_shuffle_train,
        )

        val_dataset = FileGrabber(
            files=val_source,
            patchsize=patchsize,
            in_channels=in_channels,
            out_channels=out_channels,
            dtype_overlay_out=dtype_overlay_out,
            dtype_masks_out=dtype_masks_out,
            gen_type=gen_type,
            gen_seed=gen_seed,
            num_shuffle=num_shuffle_val,
        )

        if self.DEBUG:
            print("Autobatch Data Generators created.")

        return (
            DataLoader(
                dataset=train_dataset,
                batch_size=train_batchsize,
                shuffle=train_shuffle_per_draw,
                pin_memory=pin_memory,
                num_workers=num_train_workers,
                prefetch_factor=prefetch,
                persistent_workers=persistant_workers,
            ),
            DataLoader(
                dataset=val_dataset,
                batch_size=val_batchsize,
                shuffle=val_shuffle_per_draw,
                pin_memory=pin_memory,
                num_workers=num_val_workers,
                prefetch_factor=prefetch,
                persistent_workers=persistant_workers,
            ),
        )

    # %% Helper Functions
    def __load_config__(self, config_path: str) -> None:
        with open(config_path, "r") as f:
            self.config = json.load(f)

    def __load_sources__(self) -> None:
        if self.DEBUG:
            print("Loading data sources.")

        self.__load_raw_source__()
        self.__load_val_source__()

    def __call_sources__(self) -> None:
        if self.DEBUG:
            print("Loading data sources.")

        self.__call_raw_source__()
        self.__call_val_source__()

    def __call_raw_source__(self) -> None:
        if self.DEBUG:
            print("Callling raw data source.")

        self.config["raw_source"] = pD.askDIR(
            "Please provide the path to the raw data source."
        )

    def __call_val_source__(self) -> None:
        if self.DEBUG:
            print("Callling validation data source.")

        self.config["val_source"] = pD.askDIR(
            "Please provide the path to the validation data source."
        )

    def __load_raw_source__(self) -> None:
        if self.DEBUG:
            print("Loading raw data source.")

        self.data_path_container["raw_source"] = glob.glob(
            f"{self.config['raw_source']}/*.{self.datatype}"
        )

    def __load_val_source__(self) -> None:
        if self.DEBUG:
            print("Loading validation data source.")

        self.data_path_container["val_source"] = glob.glob(
            f"{self.config['val_source']}/*.{self.datatype}"
        )

    def get_config(self) -> Dict:
        return self.config

    def pprint_config(self) -> None:
        pprint(self.config)

    def print_config(self) -> None:
        print(json.dumps(self.config, indent=4, sort_keys=False))

    def load_config(self, config_path: str = None) -> None:
        if config_path == None:
            self.__load_config__(
                pD.askFILE("Please provide the path to the configuration file.")
            )
        self.__load_config__(config_path)

    def call_sources(self) -> None:
        self.__call_sources__()

    def load_sources(self) -> None:
        self.__load_sources__()

    # %% Checkups
    def __config_startup__(self) -> None:
        if self.DEBUG:
            print("Checking for configuration startup.")

        # Ceck for the configuration file
        if self.config == None:
            if not self.__check_config__():
                self.generate_biolerplate_config_file(outpath=f"{os.getcwd()}/config")
            self.__load_config__(
                config_path=f"{os.getcwd()}/BatchDataLoader_config.json"
            )

    def __check__(self) -> bool:
        return all(
            [
                self.__check_config__(),
                self.__check_sources__(),
                self.__check_datatype__(),
            ]
        )

    def __check_config__(self) -> bool:
        if self.DEBUG:
            print("Checking configuration.")

        try:
            assert self.config != None
            assert type(self.config) == dict
            return True
        except Exception as e:
            print(e)
            return False

    def __check_sources__(self) -> bool:
        if self.DEBUG:
            print("Checking data sources.")

        try:
            assert all(
                [key in self.config.keys() for key in ["raw_source", "val_source"]]
            )
            assert type(self.config["raw_source"]) == str
            assert type(self.config["val_source"]) == str
            return True
        except Exception as e:
            print(e)
            return False

    def __check_datatype__(self) -> bool:
        if self.DEBUG:
            print("Checking datatype.")

        try:
            assert self.datatype in ["npy", "tif", "tiff", "png", "jpg", "jpeg", "bmp"]
            return True
        except Exception as e:
            print("Datatype not supported.")
            print(e)
            return False
