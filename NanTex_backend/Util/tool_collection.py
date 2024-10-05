
######################################################
#### A collection of random tools for the backend ####
######################################################

## Dependencies
import torch
import numpy as np
import os, sys, psutil
from typing import Union, Dict

## Helper Functions
def concat_dict(d:Dict)->np.ndarray:
    return np.concatenate([val for val in d.values()])

def suggest_multi_single_core(n:int)->Union[int, Dict[int, int]]:
    ## Get the number of cores
    n_cores:int = psutil.cpu_count(logical=False)
    
    if n_cores == 1:
        return n
    else:
        n_per_core = n // n_cores
        n_per_core_dict = {i: n_per_core for i in range(n_cores)}
        n_per_core_dict[n_cores-1] += n % n_cores
        return n_per_core_dict

def detect_workers()->int:
    ## Get the number of workers
    n_worker:int = psutil.cpu_count(logical=False)
    
    if n_worker < 4:
        print("Warning: The number of workers is less than 4. This may lead to significant performance degradation.")
    print(f"Number of workers: {n_worker}")
    print(f"Number of logical cores: {psutil.cpu_count(logical=True)}")
    print(f"Don't forget to close additional applications to free up more resources.")
    return n_worker
    
def detect_gpu()->bool:
    ## Check if GPU is available
    try:
        return torch.cuda.is_available()
    except:
        return False
    
def get_device()->torch.device:
    ## Get the device
    return torch.device("cuda" if detect_gpu() else "cpu")

