
######################################################
#### A collection of random tools for the backend ####
######################################################

## Dependencies
import numpy as np
from typing import Union, Dict

## Helper Functions
def concat_dict(d:Dict)->np.ndarray:
    return np.concatenate([val for val in d.values()])

## MFX Specific Functions
def L_from_pat_geo_factor(pat_geo_factor:float)->float:
    return pat_geo_factor * 360e-9

