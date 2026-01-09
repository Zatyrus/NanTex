## Dependencies
import os
import glob
import json
import psutil
import numpy as np
from pprint import pprint
from warnings import warn

from torch.utils.data import DataLoader
from typing import Tuple, Union, Dict, List, NoReturn

## Custom Dependencies
from nantex.util import pyDialogue as pyD
from nantex.batching.euthenia import Euthenia


# %% Convenience Class
class Harmonia:
    # attributes
    data_path_container: Dict[str, List[str]]
    config: Dict[str, Union[str, int, float, bool]]
    datatype: str
    DEBUG: bool

    num_possible_train_batches: int
    num_possible_val_batches: int

    # %% properties
    @property
    def data_path_container(self) -> Dict[str, List[str]]:
        return self._data_path_container

    @data_path_container.setter
    def data_path_container(self, value: Dict[str, List[str]]) -> NoReturn:
        self._data_path_container = value

    @property
    def config(self) -> Dict[str, Union[str, int, float, bool]]:
        return self._config

    @config.setter
    def config(self, value: Dict[str, Union[str, int, float, bool]]) -> NoReturn:
        self._config = value

    @property
    def datatype(self) -> str:
        return self._datatype

    @datatype.setter
    def datatype(self, value: str) -> NoReturn:
        self._datatype = value

    @property
    def DEBUG(self) -> bool:
        return self._DEBUG

    @DEBUG.setter
    def DEBUG(self, value: bool) -> NoReturn:
        self._DEBUG = value

    @property
    def num_possible_train_batches(self) -> int:
        if (
            self._config is not None
            and self._data_path_container["raw_source"] is not None
        ):
            return (
                len(self._data_path_container["raw_source"])
                * self._config["file_count_multiplier"]
            ) // self._config["train_batchsize"]
        else:
            return 0

    @num_possible_train_batches.setter
    def num_possible_train_batches(self, value: int) -> NoReturn:
        raise NotImplementedError("num_possible_train_batches is a read-only property.")

    @property
    def num_possible_val_batches(self) -> int:
        if (
            self._config is not None
            and self._data_path_container["val_source"] is not None
        ):
            return (
                len(self._data_path_container["val_source"])
                * self._config["file_count_multiplier"]
            ) // self._config["val_batchsize"]
        else:
            return 0

    @num_possible_val_batches.setter
    def num_possible_val_batches(self, value: int) -> NoReturn:
        raise NotImplementedError("num_possible_val_batches is a read-only property.")

    def __init__(
        self, config: Dict = None, datatype: str = "npy", DEBUG: bool = False
    ) -> NoReturn:
        self._config = config
        self._DEBUG = DEBUG
        self._datatype = datatype
        self._data_path_container = {"raw_source": None, "val_source": None}

        self.__post_init__()

    def __post_init__(self) -> NoReturn:
        """Post-initialization setup for the Harmonia class.

        Returns:
            NoReturn: This function does not return a value.
        """
        self.__config_startup__()
        self.__check_datatype__()
        self.__check_config__()
        self.__warn_if_file_count_modifier_larger_than_one__()

        # Check if data sources are available
        if not self.__check_sources__():
            if self._DEBUG:
                print("Data sources are not available.")
            self.__call_sources__()

    @classmethod
    def from_config(
        cls, config_file_path: str = None, datatype: str = "npy", DEBUG: str = False
    ) -> "Harmonia":
        """Create a Harmonia instance from a configuration file.

        Args:
            config_file_path (str, optional): Path to the configuration file. Defaults to None.
            datatype (str, optional): Data type of the input files. Defaults to "npy".
            DEBUG (str, optional): Debug mode flag. Defaults to False.

        Returns:
            Harmonia: An instance of the Harmonia class.
        """
        if config_file_path is None:
            config_file_path = pyD.askFILE(
                "Please provide the path to the configuration file."
            )

        # Load the configuration file
        with open(config_file_path, "r") as f:
            config = json.load(f)
        return cls(config, datatype, DEBUG)

    # %% Convenience Functions
    @staticmethod
    def generate_boilerplate_config_file(outpath: str = None) -> NoReturn:
        """Generate a boilerplate configuration file for the data generator object.

        Args:
            outpath (str, optional): Path to the output directory. Defaults to None.

        Returns:
            NoReturn: This function does not return a value.
        """
        if outpath is None:
            outpath = pyD.askDIR("Please provide the path to the output directory.")

        # set the configuration dictionary
        config = {
            "raw_source": None,  # Required
            "val_source": None,  # Required
            "train_batchsize": 32,  # Number of patches per training batch.
            "val_batchsize": 8,  # Number of patches per validation batch.
            "train_shuffle_on_draw": True,  # Shuffle training data between batches.
            "val_shuffle_on_draw": True,  # Shuffle validation data between batches.
            "num_shuffle_train": 11,  # Number of times the training data is shuffled on setup
            "num_shuffle_val": 11,  # Number of times the validation data is shuffled on setup
            "file_count_multiplier": 1,  # Multiplier to increase the effective number of files. Use with caution.
            "patchsize": (
                256,
                256,
            ),  # Patch Dimension for training (row, column) <- play with it; use a power of 2.
            "num_train_workers": 4,  # Number of workers <- run the detect_workers() function to get a recommendation.
            "num_val_workers": 2,  # Number of workers for validation <- run the detect_workers() function to get a recommendation.
            "prefetch": 8,  # Number of batches to prefetch <- based on your computational resources.
            "in_channels": 1,  # Number of input channels <- 1 for grayscale, 3 for RGB | For the sake of NanTex, we use 1. Leave it.
            "out_channels": 3,  # Number of output channels <- Number of structures to segment.
            "dtype_overlay_out": "float32",  # Data type of the output.
            "dtype_masks_out": "float32",  # Data type of the masks.
            "gen_type": "DXSM",  # Type of random number generator.
            "gen_seed": None,  # Seed for the random number generator. Reproducibility is only one integer away. :3
            "pin_memory": True,  # Flag to pin memory. <- will consume more ressources but speed-up the process.
            "persistant_workers": True,  # Flag to keep workers alive between runs. <- will consume more ressources but speed-up the process.
        }

        # Dump the configuration file
        with open(f"{outpath}/Harmonia_config.json", "w") as f:
            json.dump(config, f, indent=4, sort_keys=False)

    # %% Factory Functions
    def build(self) -> Tuple[DataLoader, DataLoader]:
        """Main function to build Raw- and Validation data Loaders.

        Returns:
            Tuple[DataLoader, DataLoader]: Raw and validation data loader objects.
        """
        # run checkups
        if self._DEBUG:
            print("Running checks.")
        try:
            assert self.__check__()
        except Exception as e:
            print(e)
            return None

        # fetch data
        if self._DEBUG:
            print("Fetching data.")
        self.__load_sources__()

        if self._DEBUG:
            print("Building Data Loader.")
        return self.__build_Harmonia__(**self._config | self._data_path_container)

    # %% Data Loader Backend
    def __build_Harmonia__(
        self,
        raw_source: str,
        val_source: str = None,
        patchsize=(256, 256),
        in_channels=1,
        out_channels=3,
        train_batchsize: int = 32,
        val_batchsize: int = 8,
        train_shuffle_on_draw: bool = True,
        val_shuffle_on_draw: bool = True,
        num_shuffle_train: int = 7,
        num_shuffle_val: int = 7,
        num_train_workers: int = 6,
        num_val_workers: int = 2,
        file_count_multiplier: int = 1,
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
            val_source (str, optional): Path to validation data.
            patchsize (tuple, optional): Set patch dimensions in px. (row,column). Defaults to (256,256).
            in_channels (int, optional): Number of input channels. Each channel is used for one feature only. (e.g. gray scale image -> 1 in-channel). Defaults to 1.
            out_channels (int, optional): Number of output channels. Is equal to the number of structures to separate. Defaults to 3.
            train_batchsize (int, optional): Number of patches per training batch. Defaults to 32.
            val_batchsize (int, optional): Number of patches per validation batch. Defaults to 8.
            train_shuffle_on_draw (bool, optional): Flag to shuffle datasets between batches. Recommended. Defaults to True.
            val_shuffle_on_draw (bool, optional): Flag to shuffle datasets between batches. Recommended. Defaults to True.
            num_shuffle_train (int, optional): Number of times the training data is shuffled on setup. Defaults to 11.
            num_shuffle_val (int, optional): Number of times the validation data is shuffled on setup. Defaults to 11.
            num_train_workers (int, optional): Number of workers for training. Defaults to 6.
            num_val_workers (int, optional): Number of workers for validation. Defaults to 2.
            file_count_multiplier (int, optional): Multiplier to increase the effective number of files. Defaults to 1.
            prefetch (int, optional): Set how many batches should be queued per worker. Defaults to 8.
            pin_memory (bool, optional): Flag to pin memory. Defaults to True.
            persistant_workers (bool, optional): Flag to keep workers alive. Defaults to True.
            dtype_overlay_out (np.dtype, optional): Data type of the output data. Defaults to np.float32.
            dtype_masks_out (np.dtype, optional): Data type of the masks. Defaults to np.uint8.
            gen_type (str, optional): Type of random number generator. Defaults to 'DXSM'.
            gen_seed (int, optional): Seed for the random number generator. Defaults to None.
        Returns:
            Tuple[DataLoader, DataLoader]: Training and validation data loaders.
        """
        if self._DEBUG:
            print("Building Data Loader.")

        if val_source is None:
            raise ValueError("Validation source must be provided.")

        ## wrap kwargs for Euthenia

        ## Return Euthenia objects for train and test data
        train_dataset = Euthenia(
            files=raw_source,
            patchsize=patchsize,
            in_channels=in_channels,
            out_channels=out_channels,
            dtype_overlay_out=dtype_overlay_out,
            dtype_masks_out=dtype_masks_out,
            gen_type=gen_type,
            gen_seed=gen_seed,
            num_shuffle=num_shuffle_train,
            file_count_multiplier=file_count_multiplier,
        )

        val_dataset = Euthenia(
            files=val_source,
            patchsize=patchsize,
            in_channels=in_channels,
            out_channels=out_channels,
            dtype_overlay_out=dtype_overlay_out,
            dtype_masks_out=dtype_masks_out,
            gen_type=gen_type,
            gen_seed=gen_seed,
            num_shuffle=num_shuffle_val,
            file_count_multiplier=file_count_multiplier,
        )

        if self._DEBUG:
            print("Autobatch Data Generators created.")

        return (
            DataLoader(
                dataset=train_dataset,
                batch_size=train_batchsize,
                shuffle=train_shuffle_on_draw,
                pin_memory=pin_memory,
                num_workers=num_train_workers,
                prefetch_factor=prefetch,
                persistent_workers=persistant_workers,
            ),
            DataLoader(
                dataset=val_dataset,
                batch_size=val_batchsize,
                shuffle=val_shuffle_on_draw,
                pin_memory=pin_memory,
                num_workers=num_val_workers,
                prefetch_factor=prefetch,
                persistent_workers=persistant_workers,
            ),
        )

    # %% Dunder Methods for Config and Sources
    def __load_config__(self, config_path: str) -> NoReturn:
        """Load configuration from a JSON file.

        Args:
            config_path (str): Path to the configuration file.

        Returns:
            NoReturn: This function does not return a value.
        """
        with open(config_path, "r") as f:
            self._config = json.load(f)

    def __load_sources__(self) -> NoReturn:
        """Load data sources for training and validation.

        Returns:
            NoReturn: This function does not return a value.
        """
        if self._DEBUG:
            print("Loading data sources.")

        self.__fetch_raw_files__()
        self.__fetch_validation_files__()

    def __call_sources__(self) -> NoReturn:
        """Call the explorer for all data sources.

        Returns:
            NoReturn: This function does not return a value.
        """
        if self._DEBUG:
            print("Loading data sources.")

        self.__call_for_raw_source__()
        self.__call_for_val_source__()

    def __call_for_raw_source__(self) -> NoReturn:
        """Call for raw data source using the explorer.

        Returns:
            NoReturn: This function does not return a value.
        """
        if self._DEBUG:
            print("Calling raw data source.")

        self._config["raw_source"] = pyD.askDIR(
            "Please provide the path to the raw data source."
        )

    def __call_for_val_source__(self) -> NoReturn:
        """Call the explorer if no validation data source is provided in the configuration.

        Returns:
            NoReturn: This function does not return a value.
        """
        if self._DEBUG:
            print("Callling validation data source.")

        self._config["val_source"] = pyD.askDIR(
            "Please provide the path to the validation data source."
        )

    def __fetch_raw_files__(self) -> NoReturn:
        """Fetch raw files and store them in the data path container.

        Returns:
            NoReturn: This function does not return a value.
        """
        if self._DEBUG:
            print("Loading raw data source.")

        self._data_path_container["raw_source"] = glob.glob(
            f"{self._config['raw_source']}/*.{self._datatype}"
        )

    def __fetch_validation_files__(self) -> NoReturn:
        """Fetch validation files and store them in the data path container.

        Returns:
            NoReturn: This function does not return a value.
        """
        if self._DEBUG:
            print("Loading validation data source.")

        self._data_path_container["val_source"] = glob.glob(
            f"{self._config['val_source']}/*.{self._datatype}"
        )

    def __config_startup__(self) -> NoReturn:
        """Load or generate configuration file.

        Returns:
            NoReturn: This function does not return a value.
        """
        if self._DEBUG:
            print("Checking for configuration startup.")

        # Check for the configuration file
        if self._config is None:
            if not self.__check_config__():
                self.generate_boilerplate_config_file(outpath=f"{os.getcwd()}/config")
            self.__load_config__(
                config_path=f"{os.getcwd()}/BatchDataLoader_config.json"
            )

    # %% Exposed Methods for Config and Sources
    def get_config(self) -> Dict:
        """Return the current configuration.

        Returns:
            Dict: The current configuration dictionary.
        """
        return self._config

    def pprint_config(self) -> NoReturn:
        """Print the configuration in a pretty format.

        Returns:
            NoReturn: This function does not return a value.
        """
        pprint(self._config)

    def print_config(self) -> NoReturn:
        """Print the configuration in a readable format.

        Returns:
            NoReturn: This function does not return a value.
        """
        print(json.dumps(self._config, indent=4, sort_keys=False))

    def load_config(self, config_path: str = None) -> NoReturn:
        """Load configuration from a file.

        Args:
            config_path (str, optional): Path to the configuration file. Defaults to None.

        Returns:
            NoReturn: This function does not return a value.
        """
        if config_path is None:
            self.__load_config__(
                pyD.askFILE("Please provide the path to the configuration file.")
            )
        self.__load_config__(config_path)

    def call_sources(self) -> NoReturn:
        """Call explorer if no data sources are provided in the configuration.

        Returns:
            NoReturn: This function does not return a value.
        """
        self.__call_sources__()

    def load_sources(self) -> NoReturn:
        """Load data sources from paths provided in the configuration.

        Returns:
            NoReturn: This function does not return a value.
        """
        self.__load_sources__()

    # %% Checkups
    def __check__(self) -> bool:
        """Run all checks.

        Returns:
            bool: True if all checks pass, False otherwise.
        """
        return all(
            [
                self.__check_config__(),
                self.__check_sources__(),
                self.__check_datatype__(),
            ]
        )

    def __check_config__(self) -> bool:
        """Check if the configuration is valid.

        Returns:
            bool: True if the configuration is valid, False otherwise.
        """
        if self._DEBUG:
            print("Checking configuration.")

        try:
            assert self._config is not None
            assert type(self._config) is dict
            return True
        except Exception as e:
            print(e)
            return False

    def __check_sources__(self) -> bool:
        """Check if the data sources are provided and available.

        Returns:
            bool: True if the data sources are valid, False otherwise.
        """
        if self._DEBUG:
            print("Checking data sources.")

        try:
            assert all(
                [key in self._config.keys() for key in ["raw_source", "val_source"]]
            )
            assert type(self._config["raw_source"]) is str
            assert type(self._config["val_source"]) is str
            assert os.path.exists(self._config["raw_source"])
            assert os.path.exists(self._config["val_source"])
            assert os.path.isdir(self._config["raw_source"])
            assert os.path.isdir(self._config["val_source"])
            assert (
                len(glob.glob(f"{self._config['raw_source']}/*.{self._datatype}")) > 0
            )
            assert (
                len(glob.glob(f"{self._config['val_source']}/*.{self._datatype}")) > 0
            )
            return True
        except Exception as e:
            print(e)
            return False

    def __check_datatype__(self) -> bool:
        """Check the data type of the input files.

        Returns:
            bool: True if the data type is supported, False otherwise.
        """
        if self._DEBUG:
            print("Checking datatype.")

        try:
            assert self._datatype in [
                "npy"
            ]  # "tif", "tiff", "png", "jpg", "jpeg", "bmp"
            return True
        except Exception as e:
            print("Datatype not supported.")
            print(e)
            return False

    def __warn_if_file_count_modifier_larger_than_one__(self) -> NoReturn:
        """Raise a warning if the file count modifier is provided in the configuration is larger than 1.

        Returns:
            NoReturn: This function does not return a value.
        """
        if self._DEBUG:
            print("Checking file count multiplier.")

        try:
            assert "file_count_multiplier" in self._config.keys()
            assert type(self._config["file_count_multiplier"]) is int
            assert self._config["file_count_multiplier"] > 0
            if self._config["file_count_multiplier"] > 1:
                if self._DEBUG:
                    print(
                        f"Warning: Detected file count multiplier of {self._config['file_count_multiplier']} > 1. This may lead to overfitting. Use with caution. It is highly recommended to generate more data instead."
                    )
                warn(
                    "Warning: Detected file count multiplier > 1. This may lead to overfitting. Use with caution. It is highly recommended to generate more data instead."
                )
        except Exception:
            if self._DEBUG:
                print("File count multiplier not provided. Setting to 1.")
            self._config["file_count_multiplier"] = 1

    # %% Helper Functions
    def detect_workers(self) -> int:
        """Detect the number of available CPU cores for multiprocessing.

        Returns:
            int: Number of available CPU cores.
        """
        ## Get the number of workers
        n_worker: int = psutil.cpu_count(logical=False)

        if n_worker < 4:
            print(
                "Warning: The number of workers is less than 4. This may lead to significant performance degradation."
            )
        print(f"Number of workers: {n_worker}")
        print(f"Number of logical cores: {psutil.cpu_count(logical=True)}")
        print(
            "Don't forget to close additional applications to free up more resources."
        )
        return n_worker
