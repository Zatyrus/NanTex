## Description: This module contains the implementation of evaluation metrics.

## Dependencies
import os
import sys
import numpy as np
import pathlib as pl
from skimage import io
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify

import sewar.full_ref as sfr

from overrides import override
from typing import List, Tuple, Union, Dict, Any, Optional, NoReturn, Callable

# for progress bar
#detect jupyter notebook
from IPython import get_ipython
import torch.utils
try:
    ipy_str = str(type(get_ipython()))
    if 'zmqshell' in ipy_str:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except:
    from tqdm import tqdm

# Custom Dependencies
from ..Util.pyDialogue import pyDialogue as pD
from ..Util.file_handler_core import FileHandlerCore
from ..data_postprocessing.oneiros import Oneiros

## Main Class
class Rhadamanthus(FileHandlerCore):
    data_paths_in:Dict[str, List[str]]
    data_in:Dict[str, np.ndarray]
    data_path_out:str
    
    DEBUG:bool
    metadata:Dict[str, Any]
    global_metrics:Dict[str, Any]
    
    oneiros:Oneiros
    
    def __init__(self,
                 num_features:int,
                 data_paths_in:Dict[str, List[str]],
                 data_path_out:str = None,
                 mode:str = 'has_ground_truth',
                 DEBUG:bool = False,
                 **kwargs:Any
                 ) -> None:
        # data variables
        self.data_paths_in = data_paths_in
        self.data_path_out = data_path_out
        self.data_in = {}  
        self.data_out = {}
        self.global_metrics = {}
        
        # model variables
        self.num_features = num_features

        # control variables
        self.DEBUG = DEBUG
        self.mode = mode
        
        # internal variables
        self.metadata = {}
        
        # oneiros
        self.oneiros = kwargs.get('oneiros', None)
        
        # Call from parent class
        self.__post_init__()
    
    @override
    def __post_init__(self)->NoReturn:
        if not self.oneiros:
            try:
                self.__load_data__()
                self.__setup_metadata__()
                
                if self.DEBUG:
                    print('Rhadamanthus Initialized...')
                
            except Exception as e:
                print(f'Error: {e}')
        else:
            try:
                self.__read_dream_memory__()
                self.__setup_metadata__()
                
                if self.DEBUG:
                    print('Rhadamanthus Initialized from Oneiros...')
            except Exception as e:
                print(f'Error: {e}')
    

    #%% MS Windows Path Handler
    @override
    @classmethod
    def from_explorer(cls, *args, **kwargs)->'Rhadamanthus':
        pass
    
    @classmethod
    def with_ground_truth(cls, **kwargs)->'Rhadamanthus':
        return cls.from_explorer(mode = 'has_ground_truth', **kwargs)
    
    @classmethod
    def without_ground_truth(cls, **kwargs)->'Rhadamanthus':
        return cls.from_explorer(mode = 'no_ground_truth', **kwargs)
    
    #%% General/LINUX Path Handler
    @override
    @classmethod
    def from_glob(cls, *args, **kwargs)->'Rhadamanthus':
        pass
    
    #%% Oneiros docked
    @classmethod
    def from_Oneiros(cls, oneiros:Oneiros, DEBUG:bool = False) -> 'Rhadamanthus':
        
        # grab data from oneiros
        return cls(num_features = oneiros.num_features,
                   data_paths_in = None,
                   data_path_out = oneiros.data_path_out,
                   DEBUG = DEBUG,
                   mode = oneiros.mode,
                   oneiros = oneiros)

    # write information to Oneiros
    def inform_Oneiros(self) -> Oneiros:
        pass
    
    #%% Metaparameter Handler
    @override
    def __setup_metadata__(self) -> NoReturn:
        self.metadata = {
            "image_quality_metrics": {
                "with_ground_truth": {
                    "MSE": True,
                    "RMSE": True,
                    "PSNR": True,
                    "SSIM": True,
                    "MSSSIM": True,
                    "UQI": True,
                    "ERGAS": True,
                    "RASE": True,
                    "SAM": True,
                    "VIF": True
                },
                "no_ground_truth": {
                    "FSIM": True,
                    "ISSM": True,
                    "UIQ": True 
                }
                },
            "patch_size": (256, 256),
            }
    #%% image quality metrics
    def judge(self) -> NoReturn:
        if self.DEBUG:
            print('Judging...')
        
        # run checks
        self.__run_checks__()
        
        
    #%% Data Handler
    def __judgement_factory__(self) -> Dict[str,Callable]:
        out:dict = []
        
        for key, flag in self.metadata['image_quality_metrics'].items():
            if not flag:
                continue
            out[key] = getattr(sfr, key.lower())
        
        return out
    
    def __judge__(self, img:np.ndarray, pred:np.ndarray) -> Dict[str, Any]:
        pass
    
    
    #%% Helper Functions
    def __read_dream_memory__(self) -> NoReturn:
        if self.DEBUG:
            print('Reading Dream Memory...')
        self.data_in = self.oneiros.data_out
        
    #%% Checks
    def __run_checks__(self) -> NoReturn:
        if self.DEBUG:
            print('Checking Data...')
        try:
            assert self.__check_data_in__()
            assert self.__check_metadata__()
        except Exception as e:
            print(f'Error: {e}')
            return
            
    def __check_metadata__(self) -> NoReturn:
        if self.DEBUG:
            print('Checking Metadata...')
        try:
            assert len(self.metadata) > 0
            return True
        except AssertionError:
            print('Metadata is empty...')
            return False
    
    def __check_data_in__(self)->bool:
        if self.DEBUG:
            print('Checking data...')
        if len(self.data_in) == 0:
            print('Error: Data not loaded. Please load the data before proceeding.')
            return False
        return True
