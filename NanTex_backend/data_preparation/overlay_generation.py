## Dependencies:
import sys
import ray
import time
import psutil
import warnings
import itertools
import webbrowser
import numpy as np
import albumentations as A
from typing import List, Tuple, Union, Dict, Any, Optional, NoReturn, Callable, Type

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
from ..Util.pyDialogue import pyDialogue as pD
from ..Util.file_handler_core import FileHandlerCore


class OVERLAY_HELPER:
    # %% Helper
    @staticmethod
    def __standardize_img__(img:np.ndarray, perform_standardization:bool) -> np.ndarray:
        if perform_standardization:
            return (img - np.mean(img)) / np.std(img)
        return img

    @staticmethod
    def __cast_output_to_dtype__(arr: np.ndarray, dtype: Type[np.number]) -> np.ndarray:
        return arr.astype(dtype)
    
    @staticmethod
    def __check_img_content__(img:np.ndarray, content_ratio:float) -> bool:
        # check how many pixels are non-zero, i.e. contain information
        if content_ratio == 0.0:
            return True
        elif content_ratio < 0:
            raise ValueError("Content ratio must be non-negative")
        content = np.count_nonzero(img) / img.size
        return content >= content_ratio
    
    @staticmethod
    def __save_patch_stack__(patch_collector: Dict[int, np.ndarray], data_path_out: str, key: str) -> None:
        for i, patch in patch_collector.items():
            if patch is not None:
                np.save(f"{data_path_out}/{key}_patch_{i}.npy", patch)
                
    @staticmethod
    def __ignore_flags__() -> List[str]:
        return ["dtype_out", "rotation", "perform_standardization", "augmentation", "patches", "patch_content_ratio", "show_pbar"]

    # %% Overlay Generation
    @staticmethod
    def __overlay__(img_list: List[np.ndarray]) -> np.ndarray:
        return np.sum(img_list, axis=0)

    @staticmethod
    def __generate_stack__(
        punchcard: Dict[str, Tuple[int, int]], data_in: Dict[str, List[np.ndarray]]
    ) -> NoReturn:
        # collect imgs
        out: List
        out = [
            data_in[key][value]
            for key, value in punchcard.items()
            if key not in OVERLAY_HELPER.__ignore_flags__()
        ]

        # overlay imgs
        out.append(
            OVERLAY_HELPER.__cast_output_to_dtype__(
                OVERLAY_HELPER.__standardize_img__(
                    OVERLAY_HELPER.__overlay__(out), punchcard["perform_standardization"]
                ), punchcard["dtype_out"]
            )
        )
        return np.stack(out, axis=0)

    @staticmethod
    def __generate_stack_rotation__(
        punchcard: Dict[str, Tuple[int, int]], data_in: Dict[str, List[np.ndarray]]
    ) -> NoReturn:
        # collect imgs
        out: List
        out = [
            data_in[key][value]
            for key, value in punchcard.items()
            if key not in OVERLAY_HELPER.__ignore_flags__()
        ]

        # rotate imgs
        out = [np.rot90(img, k=punchcard["rotation"][i]) for i, img in enumerate(out)]

        # overlay imgs
        out.append(
            OVERLAY_HELPER.__cast_output_to_dtype__(
                OVERLAY_HELPER.__standardize_img__(
                    OVERLAY_HELPER.__overlay__(out), punchcard["perform_standardization"]
                ), punchcard["dtype_out"]
            )
        )
        return np.stack(out, axis=0)
    
    @staticmethod
    def __generate_patches__(punchcard: Dict[str, Tuple[int, int]], data_in: Dict[str, List[np.ndarray]], overlay_worker: Callable) -> Dict[int,np.ndarray]:
        # Override disable auto-standardization for patch generation, because we need to apply it to the patches individually
        backup_perform_standardization = punchcard["perform_standardization"]
        punchcard["perform_standardization"] = False
        
        # generate base overlay (e.g. __generate_stack__)
        tmp = overlay_worker(punchcard=punchcard, data_in=data_in)
        img = tmp[-1,:,:] # <- overlay is always the last image in the stack
        masks = list(tmp[:-1,:,:]) # <- all other images are masks
        
        # restore perform_standardization flag
        punchcard["perform_standardization"] = backup_perform_standardization

        # generate patches
        patch_collector: Dict[int,np.ndarray] = {i:None for i in range(punchcard["patches"])}
        
        # open pbar
        if punchcard["show_pbar"]:
            patch_pbar = tqdm(total=punchcard["patches"], desc="Generating Patches...", file=sys.stdout, leave = False, miniters = 0)
        else:
            patch_pbar = None
        
        while any(v is None for v in patch_collector.values()):

            # extract patch
            augmented = punchcard["augmentation"](image=img, masks=masks)
            augmented_img = augmented["image"]
            augmented_masks = augmented["masks"]
            
            # check for content
            if not OVERLAY_HELPER.__check_img_content__(augmented_img, punchcard["patch_content_ratio"]):
                continue

            # find empty patch
            for i, patch in patch_collector.items():
                if patch is None:
                    # standardize the overlay patch and summarize to stack
                    patch_collector[i] = np.stack([*augmented_masks, OVERLAY_HELPER.__standardize_img__(augmented_img, punchcard['perform_standardization'])], axis=0)
                    break
            
            # update pbar
            if patch_pbar:
                patch_pbar.update(1)
            
        # close pbar
        if patch_pbar:
            patch_pbar.colour = "green"
            patch_pbar.close()
        
        return patch_collector

    @staticmethod
    def __generate_patch_overlay__(punchcard: Dict[str, Tuple[int, int]], data_in: Dict[str, List[np.ndarray]]) -> NoReturn:
        return OVERLAY_HELPER.__generate_patches__(punchcard=punchcard, data_in=data_in, overlay_worker=OVERLAY_HELPER.__generate_stack__)

    @staticmethod
    def __generate_patch_rotation__(punchcard: Dict[str, Tuple[int, int]], data_in: Dict[str, List[np.ndarray]]) -> NoReturn:
        return OVERLAY_HELPER.__generate_patches__(punchcard=punchcard, data_in=data_in, overlay_worker=OVERLAY_HELPER.__generate_stack_rotation__)

    @staticmethod
    @ray.remote(num_returns=1)
    def __multi_core_worker_generate_stack__(
        punchcard: Dict[str, Tuple[int, int]],
        data_in: Dict[str, List[np.ndarray]],
        data_path_out: str,
    ) -> str:
        # read punchcard
        key, punchcard = list(punchcard.items())[0]
        np.save(
            f"{data_path_out}/{key}.npy",
            OVERLAY_HELPER.__generate_stack__(punchcard=punchcard, data_in=data_in),
        )

        return key

    @staticmethod
    @ray.remote(num_returns=1)
    def __multi_core_worker_generate_stack_rotation__(
        punchcard: Dict[str, Tuple[int, int]],
        data_in: Dict[str, List[np.ndarray]],
        data_path_out: str,
    ) -> str:
        # read punchcard
        key, punchcard = list(punchcard.items())[0]
        np.save(
            f"{data_path_out}/{key}.npy",
            OVERLAY_HELPER.__generate_stack_rotation__(
                punchcard=punchcard, data_in=data_in
            ),
        )

        return key
    
    @staticmethod
    @ray.remote(num_returns=1)
    def __multi_core_worker_generate_patch_overlay__(
        punchcard: Dict[str, Tuple[int, int]],
        data_in: Dict[str, List[np.ndarray]],
        data_path_out: str,
    ) -> str:
        # read punchcard
        key, punchcard = list(punchcard.items())[0]
        OVERLAY_HELPER.__save_patch_stack__(
            patch_collector=OVERLAY_HELPER.__generate_patch_overlay__(
                punchcard=punchcard, data_in=data_in
            ),
            data_path_out=data_path_out,
            key=key
        )

        return key
    
    @staticmethod
    @ray.remote(num_returns=1)
    def __multi_core_worker_generate_patch_rotation__(
        punchcard: Dict[str, Tuple[int, int]],
        data_in: Dict[str, List[np.ndarray]],
        data_path_out: str,
    ) -> str:
        # read punchcard
        key, punchcard = list(punchcard.items())[0]
        OVERLAY_HELPER.__save_patch_stack__(
            patch_collector=OVERLAY_HELPER.__generate_patch_rotation__(
                punchcard=punchcard, data_in=data_in
            ),
            data_path_out=data_path_out,
            key=key
        )

        return key


class OverlayGenerator(FileHandlerCore):
    data_paths_in: Dict[str, List[str]]  # key: data name, value: list of paths
    data_path_out: str  # key: outpath
    data_in: Dict[str, List[np.ndarray]]  # key: data name, value: data <- img_stack
    data_punchcards: Dict[
        str, Dict[str, Tuple[int, int]]
    ]  # key: data name, value: dict of data <- input_key, mod_instructions

    mode: str
    augment: bool
    patches: int
    patchsize: Tuple[int, int]
    imagesize: Tuple[int, int]
    multi_core: bool
    sleeptime: float
    dtype_out: np.dtype
    dtype_in: np.dtype
    metadata: Dict[str, Any]
    augmentation_pipeline: A.Compose
    DEBUG: bool
    disable_auto_standardization: bool
    disable_patch_pbar: bool
    patch_content_ratio: float

    cluster_context: Optional[Any]  # ray cluster context

    # set properties
    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value
        self._update_metadata("mode", value)

    @property
    def augment(self):
        return self._augment

    @augment.setter
    def augment(self, value):
        self._augment = value
        self._update_metadata("augment", value)

    @property
    def patches(self):
        return self._patches

    @patches.setter
    def patches(self, value):
        self._patches = value
        self._update_metadata("patches", value)

    @property
    def patchsize(self):
        return self._patchsize

    @patchsize.setter
    def patchsize(self, value):
        self._patchsize = value
        self._update_metadata("patchsize", value)
        
    @property
    def imagesize(self):
        return self._imagesize

    @imagesize.setter
    def imagesize(self, value):
        self._imagesize = value
        self._update_metadata("imagesize", value)

    @property
    def multi_core(self):
        return self._multi_core

    @multi_core.setter
    def multi_core(self, value):
        self._multi_core = value
        self._update_metadata("multi_core", value)
    
    @property
    def DEBUG(self):
        return self._DEBUG

    @DEBUG.setter
    def DEBUG(self, value):
        self._DEBUG = value
        self._update_metadata("DEBUG", value)

    @property
    def sleeptime(self):
        return self._sleeptime

    @sleeptime.setter
    def sleeptime(self, value):
        self._sleeptime = value
        self._update_metadata("sleeptime", value)
        
    @property
    def dtype_out(self):
        return self._dtype_out

    @dtype_out.setter
    def dtype_out(self, value):
        self._dtype_out = value
        self._update_metadata("dtype_out", value)
        
    @property
    def dtype_in(self):
        return self._dtype_in
    
    @dtype_in.setter
    def dtype_in(self, value):
        self._dtype_in = value
        self._update_metadata("dtype_in", value)
        
    @property
    def patch_content_ratio(self):
        return self._patch_content_ratio
    @patch_content_ratio.setter
    def patch_content_ratio(self, value):
        self._patch_content_ratio = value
        self._update_metadata("patch_content_ratio", value)
        
    @property
    def disable_auto_standardization(self):
        return self._disable_auto_standardization
    @disable_auto_standardization.setter
    def disable_auto_standardization(self, value):
        self._disable_auto_standardization = value
        self._update_metadata("disable_auto_standardization", value)
        
    @property
    def disable_patch_pbar(self):
        return self._disable_patch_pbar
    @disable_patch_pbar.setter
    def disable_patch_pbar(self, value):
        self._disable_patch_pbar = value
        self._update_metadata("disable_patch_pbar", value)

    @property
    def augmentation_pipeline(self):
        return self._augmentation_pipeline
    @augmentation_pipeline.setter
    def augmentation_pipeline(self, value):
        self._augmentation_pipeline = value

    def __init__(
        self,
        data_paths_in: Dict[str, List[str]],
        data_path_out: str = None,
        mode: str = "overlay",
        augment: bool = False,
        patches: int = None,
        patchsize: Tuple[int, int] = (64, 64),
        imagesize: Tuple[int, int] = (4096, 4096),
        multi_core: bool = False,
        *args,
        **kwargs
    ) -> None:
        
        # data variables
        self.data_paths_in = data_paths_in
        self.data_path_out = data_path_out
        self.data_in = {}
        self.data_punchcards = {}
        
        # internal variables
        self.metadata = {}
        
        # populate values
        self._mode = mode
        self._augment = augment
        self._patches = patches
        self._patchsize = patchsize
        self._imagesize = imagesize
        self._multi_core = multi_core

        # handle sleep time
        self._sleeptime = 0.1
        if "sleeptime" in kwargs:
            self._sleeptime = kwargs["sleeptime"]

        # handle DEBUG
        self._DEBUG = False
        if "DEBUG" in kwargs:
            self._DEBUG = kwargs["DEBUG"]

        # handle dtype_out
        if self._patches:
            if self._DEBUG:
                print("Patch mode detected. Setting output dtype to float32.")
            self._dtype_out = np.float32
        else:
            if self._DEBUG:
                print("Full image mode detected. Setting output dtype to uint16.")
            self._dtype_out = np.uint16
        if "dtype_out" in kwargs:
            if self._DEBUG:
                print("User input detected. Overriding output dtype.")
            self._dtype_out = kwargs["dtype_out"]
            
        # handle dtype_in
        self._dtype_in = np.uint8 # default input dtype for most images
        if "dtype_in" in kwargs:
            if self._DEBUG:
                print("User input detected. Setting input dtype.")
            self._dtype_in = kwargs["dtype_in"]

        # handle augmentation pipeline
        self._augmentation_pipeline = None
        if "augmentation_pipeline" in kwargs:
            self._augmentation_pipeline = kwargs["augmentation_pipeline"]
            
        # handle disable auto-standardization
        if patches:
            if self._DEBUG:
                print("Patch mode detected. Auto-standardization enabled.")
            self._disable_auto_standardization = False
        else:
            if self._DEBUG:
                print("Full image mode detected. Auto-standardization disabled.")
            self._disable_auto_standardization = True
            
        if "disable_auto_standardization" in kwargs:
            if self._DEBUG:
                print("User input detected. Overriding auto-standardization setting.")
            self._disable_auto_standardization = kwargs["disable_auto_standardization"]
            
        # handle disable patch pbar
        self._disable_patch_pbar = False
        if "disable_patch_pbar" in kwargs:
            if self._DEBUG:
                print("User input detected. Overriding patch pbar setting.")
            self._disable_patch_pbar = kwargs["disable_patch_pbar"]
            
        # handle patch content grace interval
        self._patch_content_ratio = 0.10 # default 10% of patch must contain information
        if "patch_content_ratio" in kwargs:
            self._patch_content_ratio = kwargs["patch_content_ratio"]

        # DEBUG messages
        if self._DEBUG:
            if self._patches:
                print("Patch mode activated.")                
            if self._disable_auto_standardization:
                print("Auto-standardization disabled.")
            if self._patch_content_ratio > 0:
                print(f"Patch content grace interval set to {self._patch_content_ratio}.")
        
        # Call from parent class
        self.__post_init__()

    # post init routine
    def __post_init__(self):
        super().__post_init__()
        self.__check_input_dtype_on_startup__() # <- check if the input dtype is supported and if it matches the input images
        

    # %% classmethods
    @classmethod
    def from_explorer(cls, **kwargs) -> "OverlayGenerator":
        data_paths_in: Dict[str, List[str]] = {}
        feature_count: int = 1

        ## Feature File Selection

        while True:
            print(f"Please select files for feature #{feature_count}...")
            inp: Union[str, tuple] = pD.askFILES(
                query_title=f"Please select files for feature #{feature_count}..."
            )

            if inp == "":
                if feature_count == 1:
                    print("No features selected...")
                    break
                print("No more features to select...")
                break

            # add to dict
            data_paths_in.update({f"feature_{feature_count}": list(inp)})

            # increment feature count
            feature_count += 1

        return cls(data_paths_in=data_paths_in, **kwargs)

    @classmethod
    def from_glob(cls, *args, **kwargs) -> "OverlayGenerator":
        data_paths_in: Dict[str, List[str]] = {}

        for i, arg in enumerate(args):
            if not isinstance(arg, list):
                raise ValueError(f"Argument {i} is not a list...")
            data_paths_in.update({f"feature_{i}": arg})

        return cls(data_paths_in=data_paths_in, **kwargs)

    # %% Main Functions
    def generate_punchcards(self) -> NoReturn:
        ## Reset punchcards
        self.data_punchcards = {}

        ## generate punchcards
        if self._DEBUG:
            print("Generating punchcards...")

        if self._mode == "overlay":
            self.__generate_punchcard_overlay__()
        elif self._mode == "rotation":
            self.__generate_punchcard_overlay_rotation__()
        else:
            warnings.warn("Mode not supported yet...")
            
        # add augmentations on demand
        if self._patches:
            if self._DEBUG:
                print("Initializing patch mode...")
            if not self._augment:
                if self._DEBUG:
                    print("Augmentations disabled. Falling back to patch generation only.")
                self.__setup_patch_pipeline_no_augmentation__()
                self.__generate_punchcard_patch_augmentation__()
            elif self._augment:
                if self._DEBUG:
                    print("Initializing augmentation pipeline...")
                self.__setup_patch_pipeline_with_augmentation__()
                self.__generate_punchcard_patch_augmentation__()
                
        # add behavioral flags
        self.__update_patch_pbar_instructions__()

    def generate_overlay(
        self
    ) -> NoReturn:
        ## perform checks; see below for more info
        self.__perform_checkup__()

        ## pad input images
        self.__pad_input_imgs__()

        ## normalize input images
        self.__normalize_input_imgs__()

        ## generate punchcards
        self.generate_punchcards()

        ## generate overlay
        if self._DEBUG:
            print("Generating overlay...")

        if self._multi_core:
            if self.__check_ray_status__():
                self.__run_multi_core__()
            else:
                print("Ray not initialized...")
                print(
                    "Please run 'OverlayGenerator.setup_multi_core()' to initialize Ray..."
                )
        else:
            self.__run_single_core__()

    def setup_multi_core(
        self,
        num_cpu: int = psutil.cpu_count(logical=False),
        num_gpu: int = 0,
        launch_dashboard: bool = False,
    ) -> NoReturn:
        self.__setup_ray__(
            num_cpu=num_cpu, num_gpu=num_gpu, launch_dashboard=launch_dashboard
        )

    # %% Data Pre-Processing
    def __generate_stencil__(self) -> NoReturn:
        tmp_ovl: list = [
            range(i) for i in [len(self.data_in[key]) for key in self.data_in.keys()]
        ]
        tmp_ovl: list = list(itertools.product(*tmp_ovl))
        return tmp_ovl

    def __generate_stencil_rotation__(self) -> NoReturn:
        tmp_ovl: list = [
            range(i) for i in [len(self.data_in[key]) for key in self.data_in.keys()]
        ]
        rot_ovl: list = [range(4) for _ in range(len(self.data_in.keys()))]
        tmp_ovl += rot_ovl

        tmp_ovl: list = list(itertools.product(*tmp_ovl))
        return tmp_ovl

    def __generate_punchcard_overlay__(self) -> NoReturn:
        for punchcard in self.__generate_stencil__():
            # add file reference to punchcard
            self.data_punchcards[
                f"{len(self.data_in.keys())}_feature_{self._mode}_{punchcard}"
            ] = {
                list(self.data_in.keys())[i]: punchcard[i]
                for i in range(len(self.data_in.keys()))
            }
            # add dtype to punchcard
            self.data_punchcards[
                f"{len(self.data_in.keys())}_feature_{self._mode}_{punchcard}"
            ].update({"dtype_out": self._dtype_out})
            # add standardization to punchcard
            self.data_punchcards[
                f"{len(self.data_in.keys())}_feature_{self._mode}_{punchcard}"
            ].update({"perform_standardization": not self._disable_auto_standardization})

    def __generate_punchcard_overlay_rotation__(self) -> NoReturn:
        for punchcard in self.__generate_stencil_rotation__():
            # add file reference to punchcard
            self.data_punchcards[
                f"{len(self.data_in.keys())}_feature_{self._mode}_{punchcard}"
            ] = {
                list(self.data_in.keys())[i]: punchcard[i]
                for i in range(len(self.data_in.keys()))
            }
            # add rotation to punchcard
            self.data_punchcards[
                f"{len(self.data_in.keys())}_feature_{self._mode}_{punchcard}"
            ].update({"rotation": punchcard[len(self.data_in.keys()) :]})
            # add dtype to punchcard
            self.data_punchcards[
                f"{len(self.data_in.keys())}_feature_{self._mode}_{punchcard}"
            ].update({"dtype_out": self._dtype_out})
            # add standardization to punchcard
            self.data_punchcards[
                f"{len(self.data_in.keys())}_feature_{self._mode}_{punchcard}"
            ].update({"perform_standardization": not self._disable_auto_standardization})

    def __generate_punchcard_patch_augmentation__(self) -> NoReturn:
        # as these punchcard will always be triggered after the initial generation (e.g. overlay), we can directly reference the punchcards stored in self.data_punchcards
        for punchcard in self.data_punchcards:
            self.data_punchcards[punchcard].update(
                {"augmentation": self._augmentation_pipeline,
                 "patches": self._patches,
                 "patch_content_ratio": self._patch_content_ratio}
            )
            
    def __update_patch_pbar_instructions__(self) -> NoReturn:
        # will be triggered after punchcard generation, so we can directly reference the punchcards stored in self.data_punchcards
        if self._patches and not self._multi_core:
            for punchcard in self.data_punchcards:
                self.data_punchcards[punchcard].update(
                    {"show_pbar": not self._disable_patch_pbar}
                )
        elif self._patches and self._multi_core:
            for punchcard in self.data_punchcards:
                self.data_punchcards[punchcard].update(
                    {"show_pbar": False}
                )

    # %% Single Core Execution
    def __single_core_main__(self, stack_generation: Callable, desc: str) -> NoReturn:
        if self.patches:
            with tqdm(
                total=len(self.data_punchcards.keys()), desc=desc, file=sys.stdout, miniters = 0
            ) as pbar:
                # iterate over punchcards
                for key, punchcard in self.data_punchcards.items():
                    OVERLAY_HELPER.__save_patch_stack__(patch_collector=stack_generation(punchcard=punchcard, 
                                                                                         data_in=self.data_in), 
                                                        data_path_out=self.data_path_out, 
                                                        key=key)

                    ## update progress bar
                    pbar.update(1)

                # handle pbar
                pbar.colour = "green"
                pbar.close()
        
        else:
            with tqdm(
                total=len(self.data_punchcards.keys()), desc=desc, file=sys.stdout, miniters = 0
            ) as pbar:
                # iterate over punchcards
                for key, punchcard in self.data_punchcards.items():
                    np.save(
                        f"{self.data_path_out}/{key}.npy",
                        stack_generation(punchcard=punchcard, data_in=self.data_in),
                    )

                    ## update progress bar
                    pbar.update(1)

                # handle pbar
                pbar.colour = "green"
                pbar.close()
                

    def __run_single_core__(self) -> NoReturn:
        if self._mode == "overlay":
            if self._patches:
                self.__single_core_main__(
                    stack_generation=OVERLAY_HELPER.__generate_patch_overlay__,
                    desc="Generating Single Core Patch Overlays...",
                )
            else:
                self.__single_core_main__(
                    stack_generation=OVERLAY_HELPER.__generate_stack__,
                    desc="Generating Single Core Overlays...",
                )
        elif self._mode == "rotation":
            if self._patches:
                self.__single_core_main__(
                    stack_generation=OVERLAY_HELPER.__generate_patch_rotation__,
                    desc="Generating Single Core Patch Rotations...",
                )
            else:
                self.__single_core_main__(
                    stack_generation=OVERLAY_HELPER.__generate_stack_rotation__,
                    desc="Generating Single Core Rotational Overlays...",
                )
        else:
            warnings.warn("Mode not supported yet...")

    # %% Multi Core Execution
    def __setup_ray__(
        self, num_cpu: int, num_gpu: int, launch_dashboard: bool
    ) -> NoReturn:
        if self._DEBUG:
            print("Setting up Ray...")

        # shutdown any stray ray instances
        ray.shutdown()

        # ray init
        cluster_context = ray.init(
            num_cpus=num_cpu, num_gpus=num_gpu, ignore_reinit_error=True
        )

        # dashboard
        if launch_dashboard:
            try:
                webbrowser.get("windows-default").open(
                    f"http://{cluster_context.dashboard_url}", autoraise=True, new=2
                )
            except Exception as e:
                print(f"Error: {e}")

        if self._DEBUG:
            print("Ray setup complete...")
            print(f"Ray Dashboard: {cluster_context.dashboard_url}")
        self._ray_instance = cluster_context

    def __shutdown_ray__(self) -> NoReturn:
        if self._DEBUG:
            print("Shutting down Ray...")
        ray.shutdown()

    def __offload_punchcards_to_ray__(self) -> List[ray.ObjectRef]:
        if self._DEBUG:
            print("Offloading punchcards to Ray...")
        return [
            ray.put({key: punchcard}) for key, punchcard in self.data_punchcards.items()
        ]

    def __offload_data_in_to_ray__(self) -> List[ray.ObjectRef]:
        if self._DEBUG:
            print("Offloading data_in to Ray...")
        return ray.put(self.data_in)

    def __offload_data_path_out_to_ray__(self) -> ray.ObjectRef:
        if self._DEBUG:
            print("Offloading data_outpath to Ray...")
        return ray.put(self.data_path_out)

    def __multi_core_main__(self, worker: Callable) -> NoReturn:
        ## offload data
        data_in_ref = self.__offload_data_in_to_ray__()
        punchcard_refs = self.__offload_punchcards_to_ray__()
        data_path_out_ref = self.__offload_data_path_out_to_ray__()

        ## listen to progress
        status, finished_states = self.__listen_to_ray_progress__(
            object_references=[
                worker.remote(
                    punchcard=punchcard,
                    data_in=data_in_ref,
                    data_path_out=data_path_out_ref,
                )
                for punchcard in tqdm(
                    punchcard_refs, desc="Scheduling Workers", position=0, miniters = 0
                )
            ],
            total=len(punchcard_refs)
        )

        ## check completion
        if status:
            try:
                assert len(finished_states) == len(punchcard_refs)
                assert all(
                    [key in finished_states for key in self.data_punchcards.keys()]
                )
            except Exception as e:
                print(f"Error: {e}")

        ## Shutdown Ray
        print("Multi Core Execution Complete...")
        print("Use 'OverlayGenerator.shutdown_multi_core()' to shutdown the cluster.")

    def __run_multi_core__(self) -> NoReturn:
        ## run multi core
        if self._mode == "overlay":
            if self._patches:
                self.__multi_core_main__(
                    worker=OVERLAY_HELPER.__multi_core_worker_generate_patch_overlay__
                )
            else:
                self.__multi_core_main__(
                    worker=OVERLAY_HELPER.__multi_core_worker_generate_stack__
                )
        elif self._mode == "rotation":
            if self._patches:
                self.__multi_core_main__(
                    worker=OVERLAY_HELPER.__multi_core_worker_generate_patch_rotation__
                )
            else:
                self.__multi_core_main__(
                    worker=OVERLAY_HELPER.__multi_core_worker_generate_stack_rotation__
                )
        else:
            warnings.warn("Mode not supported yet...")

    # %% Helper Functions
    def __pad_img__(
        self, img: np.ndarray, y_lim: int, x_lim: int) -> np.ndarray:
        # override limits if imagesize is set
        if self._imagesize:
            y_lim = max(y_lim, self._imagesize[0]+1)
            x_lim = max(x_lim, self._imagesize[1]+1)

        # determine padding
        y_pad = max(0, int(np.round((y_lim - img.shape[0]) / 2)))
        x_pad = max(0, int(np.round((x_lim - img.shape[1]) / 2)))

        # setup zero array
        out = np.zeros(shape=(y_lim, x_lim), dtype=img.dtype)  # <- we retain the dtype
        # insert image
        out[y_pad : img.shape[0] + y_pad, x_pad : img.shape[1] + x_pad] = img

        # crop to limit
        if self._imagesize:
            # calculate cropping coordinates
            y_start = int(np.floor((y_lim - self._imagesize[0]) / 2))
            y_end = -(int(np.floor((y_lim - self._imagesize[0]) / 2)) + 1)
            x_start = int(np.floor((x_lim - self._imagesize[1]) / 2))
            x_end = -(int(np.floor((x_lim - self._imagesize[1]) / 2)) + 1)

            # catch rounding issues
            # rounding to the 0th decimal may introduce offsets by up to 1 pixel per limit
            if np.abs(y_lim - y_start + y_end - self._imagesize[0]) == 1:
                y_end += 1  # hence, we adjust the end coordinate
            elif np.abs(y_lim - y_start + y_end - self._imagesize[0]) == 2:
                y_start -= 1  # hence, we adjust the start coordinate
                y_end += 1  # and the end coordinate
            if np.abs(x_lim - x_start + x_end - self._imagesize[1]) == 1:
                x_end += 1
            elif np.abs(x_lim - x_start + x_end - self._imagesize[1]) == 2:
                x_start -= 1
                x_end += 1
            # in case the cropping coordinates are off by more than 2 pixels, somethings bad has happened and we raise an exception.
            if (
                np.abs(y_lim - y_start + y_end - self._imagesize[0]) > 2
                or np.abs(x_lim - x_start + x_end - self._imagesize[1]) > 2
            ):
                raise ValueError("Cropping coordinates are off by more than 2 pixels.")

            out = out[y_start:y_end, x_start:x_end]
            return out
        # return padded image
        return out

    def __normalize_img__(
        self, img: np.ndarray) -> np.ndarray:
        # if we expect to output float32 or float64, we will normalize to [0,1]
        if self._dtype_out in {np.float32, np.float64, "float32", "float64"}:
            return ((img - np.iinfo(img.dtype).min) / (np.iinfo(img.dtype).max - np.iinfo(img.dtype).min)).astype(
                self._dtype_out
            )
        # if we expect to output uint8 or uint16, we will normalize to [0,255] <- to keep compatibility
        elif self._dtype_out in {np.uint8, np.uint16, "uint8", "uint16"}:
            return ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(
                self._dtype_out
            )
        else:
            raise ValueError("Unsupported output dtype. Please choose from float32, float64, uint8, uint16.")

    def get_x_lim(self) -> int:
        return max([img.shape[1] for img in sum(list(self.data_in.values()), [])])

    def get_y_lim(self) -> int:
        return max([img.shape[0] for img in sum(list(self.data_in.values()), [])])

    def __setup_patch_pipeline_no_augmentation__(self) -> NoReturn:
        if self._DEBUG:
            print("Setting up patch pipeline without augmentation...")
        # setup patch pipeline without augmentation
        self._augmentation_pipeline = A.Compose([
            A.RandomCrop(height=self._patchsize[0], width=self._patchsize[1]),
        ])
        
    def __setup_patch_pipeline_with_augmentation__(self) -> NoReturn:
        if self._DEBUG:
            print("Setting up patch pipeline with augmentation...")
        if self._augmentation_pipeline is None:
            if self._DEBUG:
                print("No augmentation pipeline found, falling back to default...")
            self._augmentation_pipeline = A.Compose([
                A.RandomCrop(height=self._patchsize[0], width=self._patchsize[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ])
        else:
            # check the type and compose if necessary
            if type(self._augmentation_pipeline) is not A.Compose:
                if self._DEBUG:
                    print("Augmentation pipeline provided uncomposed, composing now...")
                if self.__is_iterable__(self._augmentation_pipeline):
                    if self._DEBUG:
                        print("Composing augmentation pipeline from iterable...")
                    self._augmentation_pipeline = A.Compose([*self._augmentation_pipeline])
                else:
                    if self._DEBUG:
                        print("Composing augmentation pipeline from single transform...")
                    self._augmentation_pipeline = A.Compose([self._augmentation_pipeline])
            
            # check if random crop is applied
            if self._patches:
                if not any(
                    isinstance(t, A.RandomCrop) for t in self._augmentation_pipeline.transforms
                ):
                    if self._DEBUG:
                        print("RandomCrop not found in augmentation pipeline, adding it as the first transform...")
                    self._augmentation_pipeline.transforms.insert(0, A.RandomCrop(height=self._patchsize[0], width=self._patchsize[1]))

    def __cast_to_dtype__(self, img: np.ndarray, dtype: np.dtype) -> np.ndarray:
        if img.dtype == dtype:
            if self._DEBUG:
                print(f"Type casting called unnecessarily. Image already of type {dtype}. Skipping...")
            return img

        if self._DEBUG:
            print(f"Casting image to {dtype}...")
        if np.issubdtype(img.dtype, np.floating) and np.issubdtype(dtype, np.integer):
            # float to int
            warnings.warn("Casting from float to int may lead to data loss. "
                          "Automatic normalization applied. Scaling to full range of target dtype.")
            info = np.iinfo(dtype)
            # normalize img first
            img = (img - np.min(img)) / (np.max(img) - np.min(img))  # normalize to [0,1]
            img = img * (info.max - info.min) + info.min  # scale to [min,max] of target dtype
            return img.astype(dtype)
        elif np.issubdtype(img.dtype, np.integer) and np.issubdtype(dtype, np.floating):
            pass  # int to float, no normalization needed, leaving the hook for later           
        return img.astype(dtype)
    
    def __cast_to_expected_output_dtype__(self, img: np.ndarray) -> np.ndarray:
        return self.__cast_to_dtype__(img=img, dtype=self._dtype_out)
    
    def __cast_to_expected_input_dtype__(self, img: np.ndarray) -> np.ndarray:
        return self.__cast_to_dtype__(img=img, dtype=self._dtype_in)
    
    # %% Ray 
    def __listen_to_ray_progress__(
        self,
        object_references: List[ray.ObjectRef],
        total: int,
    ) -> bool:
        if self._DEBUG:
            print("Setting up progress monitors...")
        ## create progress monitors
        ray_progress = tqdm(total=total, desc="Workers", position=1, miniters = 0)
        cpu_progress = tqdm(
            total=100,
            desc="CPU usage",
            bar_format="{desc}: {percentage:3.0f}%|{bar}|",
            position=2,
            miniters=0,
            mininterval=self._sleeptime
        )
        mem_progress = tqdm(
            total=psutil.virtual_memory().total,
            desc="RAM usage",
            bar_format="{desc}: {percentage:3.0f}%|{bar}|",
            position=3,
            miniters=0,
            mininterval=self._sleeptime
        )

        finished_states = []

        if self._DEBUG:
            print("Listening to Ray Progress...")
        ## listen for progress
        while len(object_references) > 0:
            try:
                # get the ready refs
                finished, object_references = ray.wait(object_references, timeout=8.0)

                data = ray.get(finished)
                finished_states.extend(data)

                # update the progress bars
                mem_progress.n = psutil.virtual_memory().used
                mem_progress.refresh()

                cpu_progress.n = psutil.cpu_percent()
                cpu_progress.refresh()

                # update the progress bar
                ray_progress.n = len(finished_states)
                ray_progress.refresh()

            except KeyboardInterrupt:
                print("Interrupted")
                break

        # set the progress bars to success
        ray_progress.colour = "green"
        cpu_progress.colour = "green"
        mem_progress.colour = "green"

        # set the progress bars to their final values
        ray_progress.n = total
        cpu_progress.n = 0
        mem_progress.n = 0

        # close the progress bars
        ray_progress.close()
        cpu_progress.close()
        mem_progress.close()

        if self._DEBUG:
            print("Ray Progress Complete...")

        return True, finished_states

    def __check_ray_status__(self) -> NoReturn:
        if self._DEBUG:
            print("Checking Ray Status...")
        return ray.is_initialized()

    def shutdown_multi_core(self) -> NoReturn:
        self.__shutdown_ray__()

    def __reset__(self) -> NoReturn:
        self.data_in = {}
        self.data_punchcards = {}
        self.metadata = {}
        self.data_path_out = None

        try:
            self.__shutdown_ray__()
        except Exception as e:
            print(f"Error: {e}")

    def _reboot_(self) -> NoReturn:
        self.__reset__()
        self.__post_init__()

    # %% Pre-Processing
    def __pad_input_imgs__(self) -> NoReturn:
        if self._DEBUG:
            print("Padding input images...")
        for key, value in self.data_in.items():
            self.data_in[key] = [
                self.__pad_img__(
                    img=img,
                    y_lim=self.metadata["y_lim"],
                    x_lim=self.metadata["x_lim"]
                )
                for img in value
            ]

    def __normalize_input_imgs__(self) -> NoReturn:
        if self._DEBUG:
            print("Normalizing input images...")
        for key, value in self.data_in.items():
            self.data_in[key] = [
                self.__normalize_img__(img=img) for img in value
            ]   

    # %% Metadata Handling
    def __setup_metadata__(self) -> NoReturn:
        if self._DEBUG:
            print("Setting up metadata...")
        self.metadata = {
            "x_lim": self.get_x_lim(),
            "y_lim": self.get_y_lim(),
            "dtype_out": self._dtype_out,
            "dtype_in": self._dtype_in,
            "mode": self._mode,
            "multi_core": self._multi_core,
            "patches": self._patches,
            "imagesize": self._imagesize,
            "patchsize": self._patchsize,
            "augment": self._augment,
            "DEBUG": self._DEBUG,
            "sleeptime": self._sleeptime,
            "patch_content_ratio": self._patch_content_ratio,
            "disable_patch_pbar": self._disable_patch_pbar,
            "disable_auto_standardization": self._disable_auto_standardization,
        }

    def _update_metadata(self, key: str, value: Any) -> NoReturn:
        if key in self.metadata:
            self.metadata[key] = value
        else:
            raise KeyError(f"Metadata key '{key}' not found.")

    # %% Safety checks
    def __perform_checkup__(self) -> NoReturn:
        if self._DEBUG:
            print("Performing startup checks...")
        # startup checks
        startup_check = self.__startup_check__()
        if startup_check is not True:
            raise ValueError(f"Startup check failed: {startup_check}")

        # check input dtype
        self.__check_input_dtype_on_runtime__()  # <- img is not used in the function, hence we pass None

        # check expected output type
        self.__check_expected_output__()

        # check for suboptimal dtype usage
        self.__check_suboptimal_dtype__()

        # handle disable_auto_standardization warnings
        self.__handle_disable_auto_standardization__()
    
    def __startup_check__(self) -> Union[bool, str]:
        if self._DEBUG:
            print("Starting up checks...")
        if not self.data_in:
            return "No input data found."
        if not self.metadata:
            return "Metadata not set up."
        return True

    def __check_expected_output__(self) -> NoReturn:
        if self._dtype_out not in {np.float32, np.float64, np.uint8, np.uint16, 'float32', 'float64', 'uint8', 'uint16'}:
            raise ValueError("Unsupported output dtype. Please choose from float32, float64, uint8, uint16.")
        
    def __check_suboptimal_dtype__(self) -> NoReturn:
        if self._dtype_out in {np.uint8, np.uint16, 'uint8', 'uint16'} and self._patches:
            warnings.warn("Using patch generation with output dtype uint8 or uint16 may lead to unexpected results. Consider using float32 or float64 as output dtype.")
        elif self._dtype_out in {np.float32, np.float64, 'float32', 'float64'} and not self._patches:
            warnings.warn("Using float32 or float64 output dtype with full width overlay generation will lead to significant file size increases. Consider using uint8 or uint16 as output dtype.")
            
    def __handle_disable_auto_standardization__(self) -> NoReturn:
        if self._disable_auto_standardization:
            if self._DEBUG:
                print("Auto-standardization disabled.")
            if self._dtype_out in {np.float32, np.float64, 'float32', 'float64'}:
                warnings.warn("Auto-standardization disabled while output dtype is float32 or float64.")
        elif not self._disable_auto_standardization:
            if self._DEBUG:
                print("Auto-standardization enabled.")
            if self._dtype_out in {np.uint8, np.uint16, 'uint8', 'uint16'}:
                warnings.warn("Auto-standardization enabled while output dtype is uint8 or uint16. Disabling auto-standardization to prevent unexpected results.")
                self._disable_auto_standardization = True
                
    def __check_input_dtype_on_startup__(self) -> NoReturn:
        if self._DEBUG:
            print("Checking input dtype...")
        
        # check if input dtype is supported
        assert self._dtype_in in {np.uint8, np.uint16, np.float32, np.float64, 'uint8', 'uint16', 'float32', 'float64'}, "Unsupported input dtype. Please choose from uint8, uint16, float32, float64."
            
        # collect dtype of input
        input_dtypes = {f"{feature_key}_{i}": img.dtype for feature_key, imgs in self.data_in.items() for i, img in enumerate(imgs)}
        present_dtypes = set(input_dtypes.values())
        if len(present_dtypes) > 1:
            warnings.warn(f"Multiple input dtypes found: {set(input_dtypes.values())}. Consider standardizing input dtypes.")
            # collect present dtypes and return the keys to them
            user_info = {dtype: [key for key, dt in input_dtypes.items() if dt == dtype] for dtype in present_dtypes}
            for dtype, keys in user_info.items():
                print(f" - dtype {dtype}: {keys}")
        else:
            input_dtype = present_dtypes.pop()
            if input_dtype != self._dtype_in:
                print(f"Input dtype ({input_dtype}) does not match expected dtype ({self._dtype_in}). Check your input data.")
                print(f"Overriding input dtype from {self._dtype_in} to {input_dtype}.")
                self.dtype_in = input_dtype
            else:
                if self._DEBUG:
                    print(f"Input dtype ({input_dtype}) matches expected dtype ({self._dtype_in}).")
                    
    def __check_input_dtype_on_runtime__(self) -> NoReturn:
        if self._DEBUG:
            print("Checking input dtype at runtime...")
        # iterate over collector and check dtypes, cast if required
        if not any(img.dtype != self._dtype_in for value in self.data_in.values() for img in value):
            if self._DEBUG:
                print("All input dtypes match expected dtype.")  
        else:  
            for key, value in self.data_in.items():
                if any(img.dtype != self._dtype_in for img in value):
                    if self._DEBUG:
                        print(f"Input dtype mismatch found in feature '{key}'. Casting to expected dtype '{self._dtype_in}'...")
                    self.data_in[key] = [self.__cast_to_expected_input_dtype__(img=img) for img in value]
            
        # Check input dtype after casting
        self.__check_input_dtype_on_startup__()