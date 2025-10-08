## Dependencies:
import sys
import ray
import numpy as np

## typing
from typing import List, Tuple, Dict, NoReturn, Callable, Type

## TQDM progress bar
## detect jupyter notebook
from IPython import get_ipython
try:
    ipy_str = str(type(get_ipython()))
    if "zmqshell" in ipy_str:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except Exception as e:
    print(f"Error occurred while importing tqdm: {e}")
    from tqdm import tqdm

## Class
class TekhneCore:
    # %% Helper
    @staticmethod
    def __standardize_img__(
        img: np.ndarray, perform_standardization: bool
    ) -> np.ndarray:
        if perform_standardization:
            return (img - np.mean(img)) / np.std(img)
        return img

    @staticmethod
    def __cast_output_to_dtype__(arr: np.ndarray, dtype: Type[np.number]) -> np.ndarray:
        return arr.astype(dtype)

    @staticmethod
    def __check_img_content__(img: np.ndarray, content_ratio: float) -> bool:
        # check how many pixels are non-zero, i.e. contain information
        if content_ratio == 0.0:
            return True
        elif content_ratio < 0:
            raise ValueError("Content ratio must be non-negative")
        content = np.count_nonzero(img) / img.size
        return content >= content_ratio

    @staticmethod
    def __save_patch_stack__(
        patch_collector: Dict[int, np.ndarray], data_path_out: str, key: str
    ) -> None:
        for i, patch in patch_collector.items():
            if patch is not None:
                np.save(f"{data_path_out}/{key}_patch_{i}.npy", patch)

    @staticmethod
    def __ignore_flags__() -> List[str]:
        return [
            "dtype_out",
            "rotation",
            "perform_standardization",
            "augmentation",
            "patches",
            "patch_content_ratio",
            "show_pbar",
        ]

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
            if key not in TekhneCore.__ignore_flags__()
        ]

        # overlay imgs
        out.append(
            TekhneCore.__cast_output_to_dtype__(
                TekhneCore.__standardize_img__(
                    TekhneCore.__overlay__(out),
                    punchcard["perform_standardization"],
                ),
                punchcard["dtype_out"],
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
            if key not in TekhneCore.__ignore_flags__()
        ]

        # rotate imgs
        out = [np.rot90(img, k=punchcard["rotation"][i]) for i, img in enumerate(out)]

        # overlay imgs
        out.append(
            TekhneCore.__cast_output_to_dtype__(
                TekhneCore.__standardize_img__(
                    TekhneCore.__overlay__(out),
                    punchcard["perform_standardization"],
                ),
                punchcard["dtype_out"],
            )
        )
        return np.stack(out, axis=0)

    @staticmethod
    def __generate_patches__(
        punchcard: Dict[str, Tuple[int, int]],
        data_in: Dict[str, List[np.ndarray]],
        overlay_worker: Callable,
    ) -> Dict[int, np.ndarray]:
        # Override disable auto-standardization for patch generation, because we need to apply it to the patches individually
        backup_perform_standardization = punchcard["perform_standardization"]
        punchcard["perform_standardization"] = False

        # generate base overlay (e.g. __generate_stack__)
        tmp = overlay_worker(punchcard=punchcard, data_in=data_in)
        img = tmp[-1, :, :]  # <- overlay is always the last image in the stack
        masks = list(tmp[:-1, :, :])  # <- all other images are masks

        # restore perform_standardization flag
        punchcard["perform_standardization"] = backup_perform_standardization

        # generate patches
        patch_collector: Dict[int, np.ndarray] = {
            i: None for i in range(punchcard["patches"])
        }

        # open pbar
        if punchcard["show_pbar"]:
            patch_pbar = tqdm(
                total=punchcard["patches"],
                desc="Generating Patches...",
                file=sys.stdout,
                leave=False,
                miniters=0,
            )
        else:
            patch_pbar = None

        while any(v is None for v in patch_collector.values()):
            # extract patch
            augmented = punchcard["augmentation"](image=img, masks=masks)
            augmented_img = augmented["image"]
            augmented_masks = augmented["masks"]

            # check for content
            if not TekhneCore.__check_img_content__(
                augmented_img, punchcard["patch_content_ratio"]
            ):
                continue

            # find empty patch
            for i, patch in patch_collector.items():
                if patch is None:
                    # standardize the overlay patch and summarize to stack
                    patch_collector[i] = np.stack(
                        [
                            *augmented_masks,
                            TekhneCore.__standardize_img__(
                                augmented_img, punchcard["perform_standardization"]
                            ),
                        ],
                        axis=0,
                    )
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
    def __generate_patch_overlay__(
        punchcard: Dict[str, Tuple[int, int]], data_in: Dict[str, List[np.ndarray]]
    ) -> NoReturn:
        return TekhneCore.__generate_patches__(
            punchcard=punchcard,
            data_in=data_in,
            overlay_worker=TekhneCore.__generate_stack__,
        )

    @staticmethod
    def __generate_patch_rotation__(
        punchcard: Dict[str, Tuple[int, int]], data_in: Dict[str, List[np.ndarray]]
    ) -> NoReturn:
        return TekhneCore.__generate_patches__(
            punchcard=punchcard,
            data_in=data_in,
            overlay_worker=TekhneCore.__generate_stack_rotation__,
        )

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
            TekhneCore.__generate_stack__(punchcard=punchcard, data_in=data_in),
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
            TekhneCore.__generate_stack_rotation__(
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
        TekhneCore.__save_patch_stack__(
            patch_collector=TekhneCore.__generate_patch_overlay__(
                punchcard=punchcard, data_in=data_in
            ),
            data_path_out=data_path_out,
            key=key,
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
        TekhneCore.__save_patch_stack__(
            patch_collector=TekhneCore.__generate_patch_rotation__(
                punchcard=punchcard, data_in=data_in
            ),
            data_path_out=data_path_out,
            key=key,
        )

        return key
