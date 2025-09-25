## Description: This module contains the implementation of evaluation metrics.

## Dependencies
import torch
import numpy as np
from psutil import cpu_count
from ezRay import MultiCoreExecutionTool

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
    data_path_out: str

    data_in: Dict[str, np.ndarray]
    results: Dict[str, Any]
    metadata: Dict[str, Any]
    metrics: Dict[str, Any]

    DEBUG: bool
    disable_tqdm: bool
    num_features: int

    oneiros: Oneiros
    multi_cpu_tool: MultiCoreExecutionTool
    
    mode: str  # has_ground_truth or no_ground_truth
    device: str = "cpu"  # or gpu or multi-cpu if available
    patchsize: Tuple[int, int]  # default patch size


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

                # handle multi-cpu setup
                if self.device == "multi_cpu":
                    self.__startup_multi_cpu__()

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
        
        # normalize data
        self.__min_max_normalize_data__()

        # cast to tensor
        self.__cast_to_tensor__()

        # assure shapes
        self.__ensure_shape__()
        
        # send to device
        self.__send_to_device__()

        # map to all data
        match self.device:
            case "cpu":
                self.results = {
                    key: self.__judge_cpu__(**data)
                    for key, data in tqdm(
                        self.data_in.items(),
                        disable=self.disable_tqdm,
                        desc="Evaluating CPU...",
                    )
                }
            case "cuda":
                self.results = {
                    key: self.__judge_cuda__(**data)
                    for key, data in tqdm(
                        self.data_in.items(),
                        disable=self.disable_tqdm,
                        desc="Evaluating CUDA...",
                    )
                }
            case "multi_cpu":
                # upload data to multi-cpu tool
                self.__push_data_to_multi_cpu__()

                # evaluate using multi-cpu tool
                self.results = {
                    k: v["result"] for k, v in self.multi_cpu_tool.run(Rhadamanthus.__judge_multi_cpu__).items()
                }
            case _:
                pass

    # %% Data Handler
    @staticmethod
    def __base_judgement_worker__(
        metrics: Dict[str, Any], img: torch.Tensor, pred: torch.Tensor
    ) -> Generator:
        for method_key, method in metrics.items():
            if "update" in dir(method):
                method.update(img, pred)
                yield method_key, method.compute().item()
            else:
                yield method_key, method(img, pred).item()

    @staticmethod
    def __judgement__(
        judgement_worker: Generator,
        metrics: Dict[str, Any],
        features: Dict[str, torch.Tensor],
        dreams: Dict[str, torch.Tensor],
    ) -> Generator:
        for i in range(len(features.keys())):
            yield (
                f"feature_{i}",
                {
                    k: v
                    for k, v in judgement_worker(
                        metrics, features[f"feature_{i}"], dreams[f"dream_{i}"]
                    )
                },
            )

    def __judge_cpu__(
        self, features: torch.Tensor, dreams: torch.Tensor
    ) -> Dict[str, Any]:
        return dict(
            Rhadamanthus.__judgement__(
                Rhadamanthus.__base_judgement_worker__, self.metrics, features, dreams
            )
        )

    def __judge_cuda__(self) -> NoReturn:
        raise NotImplementedError("CUDA evaluation not implemented yet.")

    @staticmethod
    def __judge_multi_cpu__(**kwargs) -> Dict[str, Dict[str, Any]]:
        return dict(Rhadamanthus.__judgement__(**kwargs))

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

    def __send_to_device__(self) -> NoReturn:
        if self.DEBUG:
            print("Sending data to device...")
        match self.device:
            case "cpu":
                for key, data in self.data_in.items():
                    self.data_in[key] = {
                        "features": {
                            k: v.to("cpu") for k, v in data["features"].items()
                        },
                        "dreams": {k: v.to("cpu") for k, v in data["dreams"].items()},
                    }
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

    def __startup_multi_cpu__(self, **kwargs) -> NoReturn:
        if self.DEBUG:
            print("Starting up Multi-CPU tool...")

        ## Default Cluster Settings
        instance_metadata: dict = {"num_cpus": 1, "num_gpus": 0, "ignore_reinit_error": True}
        if "instance_metadata" in kwargs:
            instance_metadata.update(kwargs.get("instance_metadata", {}))

        ## Default Task Settings
        task_metadata: dict = {"num_cpus": 1, "num_gpus": 0, "num_returns": 1}
        if "task_metadata" in kwargs:
            task_metadata.update(kwargs.get("task_metadata", {}))

        ## Default Verbosity Settings
        verbosity_flags: dict = {
            "AutoArchive": False,
            "AutoContinue": True,
            "SingleShot": True,
            "AutoLaunchDashboard": False,
            "silent": False,
            "DEBUG": False,
        }
        if "verbosity_flags" in kwargs:
            verbosity_flags.update(kwargs.get("verbosity_flags", {}))

        ## Assembly
        RuntimeMetadata: Dict[str, Any] = {
            "instance_metadata": instance_metadata,
            "task_metadata": task_metadata,
        }

        # initialize tool
        self.multi_cpu_tool = MultiCoreExecutionTool(
            **RuntimeMetadata, **verbosity_flags,
        )

    def __push_data_to_multi_cpu__(self) -> NoReturn:
        if self.DEBUG:
            print("Pushing data to Multi-CPU tool...")
        pushed_data = self.data_in
        # add metrics and the worker function to each case
        for key in pushed_data.keys():
            pushed_data[key].update(
                {
                    "metrics": self.metrics,
                    "judgement_worker": Rhadamanthus.__base_judgement_worker__,
                }
            )
        self.multi_cpu_tool.update_data(pushed_data)

    def __min_max_normalize__(self, data: np.ndarray) -> np.ndarray:
        return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    
    def __z_score_normalize__(self, data: np.ndarray) -> np.ndarray:
        return (data - np.mean(data)) / (np.std(data) + 1e-8)
    
    def __min_max_normalize_data__(self) -> NoReturn:
        if self.DEBUG:
            print("Min-Max Normalizing data...")
        for key, data in self.data_in.items():
            if isinstance(data, dict) and "features" in data and "dreams" in data:
                self.data_in[key] = {
                    "features": {
                        k: self.__min_max_normalize__(v).astype(np.float32)
                        for k, v in data["features"].items()
                    },
                    "dreams": {
                        k: self.__min_max_normalize__(v).astype(np.float32)
                        for k, v in data["dreams"].items()
                    },
                }
            else:
                raise ValueError(f"Data format for key {key} is incorrect.")
            
    def __z_score_normalize_data__(self) -> NoReturn:
        if self.DEBUG:
            print("Z-Score Normalizing data...")
        for key, data in self.data_in.items():
            if isinstance(data, dict) and "features" in data and "dreams" in data:
                self.data_in[key] = {
                    "features": {
                        k: self.__z_score_normalize__(v) 
                        for k, v in data["features"].items()
                    },
                    "dreams": {
                        k: self.__z_score_normalize__(v) 
                        for k, v in data["dreams"].items()
                    },
                }
            else:
                raise ValueError(f"Data format for key {key} is incorrect.")

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

    # %% Exposed Methods
    def launch_multi_cpu_dashboard(self) -> NoReturn:
        if self.DEBUG:
            print("Launching Multi-CPU Dashboard...")
        try:
            self.multi_cpu_tool.launch_dashboard()
        except Exception as e:
            print(f"Error: {e}")
            return
