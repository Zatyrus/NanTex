## Description: This module contains the implementation of evaluation metrics.

## Dependencies
import torch
import numpy as np
from psutil import cpu_count

# eval method import

from overrides import override
from typing import List, Dict, Any, NoReturn, Generator, Tuple

# for progress bar
# detect jupyter notebook
from IPython import get_ipython

try:
    ipy_str = str(type(get_ipython()))
    if "zmqshell" in ipy_str:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except:
    from tqdm import tqdm

# Custom Dependencies
from ..Util.file_handler_core import FileHandlerCore
from ..data_postprocessing.Oneiros import Oneiros
from ..Util.pyDialogue import pyDialogue as pD


## Main Class
class Rhadamanthus(FileHandlerCore):
    data_paths_in: Dict[str, List[str]]
    data_in: Dict[str, np.ndarray]
    results: Dict[str, Any]
    data_path_out: str

    DEBUG: bool
    disable_tqdm: bool
    metadata: Dict[str, Any]
    metrics: Dict[str, Any]
    mode: str  # has_ground_truth or no_ground_truth
    num_features: int
    patchsize: Tuple[int, int]  # default patch size

    oneiros: Oneiros
    device: str = "cpu"  # or gpu if available

    def __init__(
        self,
        num_features: int,
        data_paths_in: Dict[str, List[str]],
        data_path_out: str = None,
        mode: str = "has_ground_truth",
        DEBUG: bool = False,
        **kwargs: Any,
    ) -> None:
        # data variables
        self.data_paths_in = data_paths_in
        self.data_path_out = data_path_out
        self.data_in = {}
        self.results = {}

        # metrics callables
        self.metrics = {}

        # model variables
        self.num_features = num_features

        # control variables
        self.DEBUG = DEBUG
        self.mode = mode

        # internal variables
        self.metadata = {}

        # oneiros hook
        self.oneiros = None

        # oneiros
        if "oneiros" in kwargs:
            if self.DEBUG:
                print("Oneiros detected.")
            self.oneiros = kwargs.get("oneiros", None)

        # Device
        if "device" in kwargs:
            if self.DEBUG:
                print(f"Trying to set device to {kwargs.get('device')}.")
            if kwargs.get("device") in ["gpu", "cuda"]:
                import torch

                if torch.cuda.is_available():
                    kwargs["device"] = "cuda"
                else:
                    if self.DEBUG:
                        print("GPU not available, defaulting to CPU.")
                    kwargs["device"] = "cpu"

            if kwargs.get("device") in ["multi_cpu"]:
                if cpu_count(logical=False) > 2:
                    self.device = "multi_cpu"
                else:
                    if self.DEBUG:
                        print(
                            "Multiple CPU cores not available, defaulting to single CPU."
                        )
                    kwargs["device"] = "cpu"

        # handle patchsize
        self.patchsize = (256, 256)  # default patch size
        if "patchsize" in kwargs:
            if (
                isinstance(kwargs.get("patchsize"), (list, tuple))
                and len(kwargs.get("patchsize")) == 2
            ):
                self.patchsize = tuple(kwargs.get("patchsize"))

        # handle disable_tqdm
        self.disable_tqdm = False
        if "disable_tqdm" in kwargs:
            if isinstance(kwargs.get("disable_tqdm"), bool):
                self.disable_tqdm = kwargs.get("disable_tqdm")

        # Call from parent class
        self.__post_init__()

    @override
    def __post_init__(self) -> NoReturn:
        if not self.oneiros:
            try:
                self.__load_data__()
                self.__setup_metadata__()
                self.__arrange_data__()  # <- separate feature and dream data

                if self.DEBUG:
                    print("Rhadamanthus Initialized...")

            except Exception as e:
                print(f"Error: {e}")
        else:
            try:
                self.__read_dream_memory__()
                self.__setup_metadata__()

                if self.DEBUG:
                    print("Rhadamanthus Initialized from Oneiros...")
            except Exception as e:
                print(f"Error: {e}")

    # %% MS Windows Path Handler
    @classmethod
    def from_explorer(cls, *args, **kwargs) -> "Rhadamanthus":
        ## initialize
        data_paths_in: Dict[str, List[str]] = {}

        ## check mode
        try:
            assert isinstance(kwargs.get("mode"), str)
            assert kwargs.get("mode") in ["has_ground_truth", "no_ground_truth"]
        except Exception as e:
            print(f"Error: {e}")
            print("Please select a valid mode from the following:")
            print("1. has_ground_truth")
            print("2. no_ground_truth")
            return None

        ## get data paths
        files_tmp = pD.askFILES(query_title="Please select the data files")
        data_paths_in.update({f"case_{i}": files_tmp[i] for i in range(len(files_tmp))})

        return cls(data_paths_in=data_paths_in, **kwargs)

    @classmethod
    def with_ground_truth(cls, **kwargs) -> "Rhadamanthus":
        return cls.from_explorer(mode="has_ground_truth", **kwargs)

    @classmethod
    def without_ground_truth(cls, **kwargs) -> "Rhadamanthus":
        return cls.from_explorer(mode="no_ground_truth", **kwargs)

    # %% General/LINUX Path Handler
    @classmethod
    def from_glob(cls, *args, **kwargs) -> "Rhadamanthus":
        pass

    # %% Oneiros docked
    @classmethod
    def from_Oneiros(cls, oneiros: Oneiros, DEBUG: bool = False) -> "Rhadamanthus":
        # grab data from oneiros
        return cls(
            num_features=oneiros.num_features,
            data_paths_in=None,
            data_path_out=oneiros.data_path_out,
            DEBUG=DEBUG,
            mode=oneiros.mode,
            oneiros=oneiros,
        )

    # write information to Oneiros
    def inform_Oneiros(self) -> Oneiros:
        pass

    # %% Metaparameter Handler
    @override
    def __setup_metadata__(self) -> NoReturn:
        self.metadata = {
            "image_quality_metrics": {},
            "patch_size": (256, 256),
        }

    # %% Main Evaluation Loop
    def judge(self) -> NoReturn:
        if self.DEBUG:
            print("Judging...")

        # run checks
        self.__run_checks__()

        # cast to tensor
        self.__cast_to_tensor__()

        # assure shapes
        self.__ensure_shape__()

        # map to all data
        match self.device:
            case "cpu":
                self.results = {
                    key: self.__judge_cpu__(**data)
                    for key, data in tqdm(self.data_in.items(), disable=self.disable_tqdm, desc="Evaluating CPU...")
                }
            case "cuda":
                self.results = {
                    key: self.__judge_cuda__(**data)
                    for key, data in tqdm(self.data_in.items(), disable=self.disable_tqdm, desc="Evaluating CUDA...")
                }
            case "multi_cpu":
                self.results = {
                    key: self.__judge_multi_cpu__(**data)
                    for key, data in tqdm(self.data_in.items(), disable=self.disable_tqdm, desc="Evaluating Multi-CPU...")
                }
            case _:
                pass

    # %% Data Handler
    def __judgement_factory__(self, img: torch.Tensor, pred: torch.Tensor) -> Generator:
        for method_key, method in self.metrics.items():
            if "update" in dir(method):
                method.update(img, pred)
                yield method_key, method.compute().item()
            else:
                yield method_key, method(img, pred)

    def __judge_base__(self, features: torch.Tensor, dreams: torch.Tensor) -> Generator:
        for i in range(self.num_features):
            yield (
                f"feature_{i}",
                {
                    k: v
                    for k, v in self.__judgement_factory__(
                        features[f"feature_{i}"], dreams[f"dream_{i}"]
                    )
                },
            )

    def __judge_cpu__(
        self, features: torch.Tensor, dreams: torch.Tensor
    ) -> Dict[str, Any]:
        return dict(self.__judge_base__(features, dreams))

    def __judge_cuda__(self) -> NoReturn:
        raise NotImplementedError("CUDA evaluation not implemented yet.")

    def __judge_multi_cpu__(self) -> NoReturn:
        raise NotImplementedError("Multi-CPU evaluation not implemented yet.")

    # %% Evaluation Metrics
    def __MSE__(self, img: np.ndarray, pred: np.ndarray) -> float:
        pass

    def __PSNR__(self, img: np.ndarray, pred: np.ndarray) -> float:
        pass

    def __SSIM__(self, img: np.ndarray, pred: np.ndarray) -> float:
        pass

    def __MSSSIM__(self, img: np.ndarray, pred: np.ndarray) -> float:
        pass

    # %% Helper Functions
    def __read_dream_memory__(self) -> NoReturn:
        if self.DEBUG:
            print("Reading Dream Memory...")
        self.data_in = self.oneiros.data_out

    def __arrange_data__(self) -> NoReturn:
        if self.DEBUG:
            print("Formatting Data...")
        for key, data in self.data_in.items():
            if isinstance(data, np.ndarray):
                self.data_in[key] = {
                    "features": {
                        f"feature_{i}": data[i] for i in range(self.num_features)
                    },
                    "dreams": {
                        f"dream_{i}": data[i + self.num_features]
                        for i in range(0, self.num_features)
                    },
                }
            elif isinstance(data, dict):
                if "features" in data and "dreams" in data:
                    continue
                else:
                    raise ValueError(f"Data format for key {key} is incorrect.")

    def __cast_to_tensor__(self) -> NoReturn:
        if self.DEBUG:
            print("Casting data to tensors...")
        for key, data in self.data_in.items():
            if isinstance(data, dict) and "features" in data and "dreams" in data:
                if all(
                    isinstance(v, torch.Tensor) for v in data["features"].values()
                ) and all(isinstance(v, torch.Tensor) for v in data["dreams"].values()):
                    continue
                elif all(
                    isinstance(v, np.ndarray) for v in data["features"].values()
                ) and all(isinstance(v, np.ndarray) for v in data["dreams"].values()):
                    self.data_in[key] = {
                        "features": {
                            k: torch.from_numpy(v) for k, v in data["features"].items()
                        },
                        "dreams": {
                            k: torch.from_numpy(v) for k, v in data["dreams"].items()
                        },
                    }
                else:
                    raise ValueError(f"Data format for key {key} is incorrect.")
            else:
                raise ValueError(f"Data format for key {key} is incorrect.")

    def __ensure_shape__(self) -> NoReturn:
        if self.DEBUG:
            print("Assuring data shapes...")
        for key, data in self.data_in.items():
            if isinstance(data, dict) and "features" in data and "dreams" in data:
                if not any(
                    v.shape == self.__expected_input_dim__()
                    for v in data["features"].values()
                ) and not any(
                    v.shape == self.__expected_input_dim__()
                    for v in data["dreams"].values()
                ):
                    self.data_in[key] = {
                        "features": {
                            k: v.unsqueeze(0).unsqueeze(0)
                            for k, v in data["features"].items()
                        },
                        "dreams": {
                            k: v.unsqueeze(0).unsqueeze(0)
                            for k, v in data["dreams"].items()
                        },
                    }
            else:
                raise ValueError(f"Data format for key {key} is incorrect.")

    def __sent_to_device__(self) -> NoReturn:
        if self.DEBUG:
            print("Sending data to device...")
        match self.device:
            case "cpu":
                pass
            case "cuda":
                for key, data in self.data_in.items():
                    self.data_in[key] = {
                        "features": {
                            k: v.to("cuda") for k, v in data["features"].items()
                        },
                        "dreams": {k: v.to("cuda") for k, v in data["dreams"].items()},
                    }
            case "multi_cpu":
                pass
            case _:
                pass

    def __expected_input_dim__(self) -> Tuple[int, int, int]:
        return (1, 1, *self.patchsize)  # (C, B, H, W)

    # %% Checks
    def __run_checks__(self) -> NoReturn:
        if self.DEBUG:
            print("Checking Data...")
        try:
            assert self.__check_data_in__()
            assert self.__check_metadata__()
        except Exception as e:
            print(f"Error: {e}")
            return

    def __check_metadata__(self) -> NoReturn:
        if self.DEBUG:
            print("Checking Metadata...")
        try:
            assert len(self.metadata) > 0
            return True
        except AssertionError:
            print("Metadata is empty...")
            return False

    def __check_data_in__(self) -> bool:
        if self.DEBUG:
            print("Checking data...")
        if len(self.data_in) == 0:
            print("Error: Data not loaded. Please load the data before proceeding.")
            return False
        return True
