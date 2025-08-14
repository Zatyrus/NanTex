import numpy as np
from Util.pyDialogue import *

from image_similarity_measures.quality_metrics import fsim, ssim, psnr
from sewar import full_ref as sfr

import ray
import pickle
import os


def evaluate():
    path = askFILES()
    results = {}

    if len(path) == 1:
        file = np.load(path[0]).astype(np.uint8)
        num_features = file.shape[2] // 2

        for i in range(num_features):
            results[f"feature_{i}"] = {
                "SSIM": [],
                "SSIM_sewar": [],
                "MSSSIM": [],
                "FSIM": [],
                "ISSM": [],
                "UIQ": [],
                "SAM": [],
            }

        eval_cases = [
            np.dstack([file[..., i], file[..., i + int(num_features)]])
            for i in range(num_features)
        ]

        for i, eval_case in enumerate(eval_cases):
            # print(i)
            # print(eval_case.shape)
            # start = time()
            # print('their_SSIM')
            results[f"feature_{i}"]["SSIM"] = ssim(
                eval_case[..., 0][..., None], eval_case[..., 1][..., None], 255
            )
            # print('their_SSIM', str(-start+time()))
            results[f"feature_{i}"]["SSIM_sewar"] = sfr.ssim(
                eval_case[..., 0][..., None], eval_case[..., 1][..., None], MAX=255
            )
            # print('sewar_SSIM', str(-start+time()))
            results[f"feature_{i}"]["MSSSIM"] = sfr.msssim(
                eval_case[..., 0][..., None], eval_case[..., 1][..., None], MAX=255
            )
            # print('sewar_MSSSIM', str(-start+time()))
            results[f"feature_{i}"]["FSIM"] = fsim(
                eval_case[..., 0][..., None], eval_case[..., 1][..., None]
            )
            # print('their_FSIM', str(-start+time()))
            # results[f'feature_{i}']['ISSM'] = issm(eval_case[...,0][...,None],eval_case[...,1][...,None])
            # print('their_ISSM', str(-start+time()))
            # #results[f'feature_{i}']['UIQ'] = uiq(eval_case[...,0][...,None],eval_case[...,1][...,None])
            # #results[f'feature_{i}']['SAM'] = sam(eval_case[...,0][...,None],eval_case[...,1][...,None])

        return results


@ray.remote
def bulk_eval(in_path, out_path):
    results = {}

    file = np.load(in_path).astype(np.uint8)
    num_features = file.shape[2] // 2

    for i in range(num_features):
        results[f"feature_{i}"] = {"PSNR": [], "SSIM": [], "MSSSIM": []}

    eval_cases = [
        np.dstack([file[..., i], file[..., i + int(num_features)]])
        for i in range(num_features)
    ]

    for i, eval_case in enumerate(eval_cases):
        results[f"feature_{i}"]["PSNR"] = psnr(
            eval_case[..., 0][..., None], eval_case[..., 1][..., None], 255
        )
        results[f"feature_{i}"]["SSIM"] = ssim(
            eval_case[..., 0][..., None], eval_case[..., 1][..., None], 255
        )
        results[f"feature_{i}"]["MSSSIM"] = sfr.msssim(
            eval_case[..., 0][..., None], eval_case[..., 1][..., None], MAX=255
        )

    with open(
        f"{out_path}/{os.path.basename(in_path)[:-8]}_eval_dump.pickle", "wb"
    ) as current:
        pickle.dump(results, current, protocol=pickle.HIGHEST_PROTOCOL)
