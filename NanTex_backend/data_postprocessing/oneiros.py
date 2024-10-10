## Dependencies
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify

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

## main class
class Oneiros(FileHandlerCore):
    data_paths_in:Dict[str, List[str]]
    data_in:Dict[str, np.ndarray]
    data_path_out:str
    data_out:Dict[str, np.ndarray]
    
    DEBUG:bool
    mode:str
    metadata:Dict[str, Any]
    data_type:str
    
    model:Optional[torch.nn.Module]
    
    def __init__(self,
                 data_paths_in:Dict[str, List[str]],
                 data_path_out:str,
                 mode:str,
                 DEBUG:bool = False
                 ) -> None:
        # data variables
        self.data_paths_in = data_paths_in
        self.data_path_out = data_path_out
        self.data_in = {}  
        self.data_out = {}
        
        # control variables
        self.DEBUG = DEBUG
        self.mode = mode
        
        # internal variables
        self.metadata = {}
        
        # DL variables
        self.model = None
        
        # Call from parent class
        self.__post_init__()
        
    def __setup_metadata__(self)->NoReturn:
        if self.DEBUG:
            print('Setting up metadata...')
        self.metadata.update(
            {
                "feature_cleanup_threshodls" : {
                    "feature_1": 0.1,
                    "feature_2": 0.1,
                    "feature_3": 0.1
                    },
                "dynamic_thresholds": {
                    "upper" : 3,
                    "lower" : -2
                    },
                "patch_size": (256, 256),
                "standardize": True,
                "normalize": False,
                "tensortype": torch.float32
            }
        )
    
    #%% Classmethods
    @classmethod
    def from_explorer(cls, mode:str, **kwargs)->'Oneiros':
        ## initialize
        data_paths_in:Dict[str, List[str]] = {}
        
        ## check mode
        try:
            assert isinstance(mode, str)
            assert mode in ['has_ground_truth', 'no_ground_truth']
        except Exception as e:
            print(f'Error: {e}')
            print('Please select a valid mode from the following:')
            print('1. has_ground_truth')
            print('2. no_ground_truth')
            return None
        
        ## get data paths
        data_paths_in['data'] = pD.askFILES(query_title = "Please select the data files")

        return cls(data_paths_in = data_paths_in, 
                   mode = mode, **kwargs)
        
    @classmethod
    def with_ground_truth(cls, **kwargs)->'Oneiros':
        return cls.from_explorer(mode = 'has_ground_truth', **kwargs)
    
    @classmethod
    def without_ground_truth(cls, **kwargs)->'Oneiros':
        return cls.from_explorer(mode = 'no_ground_truth', **kwargs)
    
    #%% Prediction Utils
    def __setup_model__(self,
                        model:torch.nn.Sequential,
                        activation:torch.nn.Module,
                        device:torch.device
                        )->NoReturn:
        if self.DEBUG:
            print('Setting up model...')
                    
        ## set model
        self.model = model
        
        ## set activation
        self.activation = activation
        
        ## set device
        self.device = device
        
        ## set model to eval
        self.model.eval()
        
        ## send model to device
        self.model.to(self.device)
    
    def __fetch_weights__(self,
                         model:torch.nn.Sequential,
                         state_dict_path:str
                         )->torch.nn.Sequential:
        if self.DEBUG:
            print('Fetching weights...')
        ## check path
        state_dict_path = self.__check_filepath__(state_dict_path, 'state_dict')
        
        ## get weights
        model.load_state_dict(torch.load(state_dict_path))
        
        return model
    
    #%% Fronend Functions
    def setup_model(self,
                    model:torch.nn.Sequential,
                    activation:torch.nn.Module,
                    device:torch.device,
                    state_dict_path:Optional[str] = None
                    )->NoReturn:
        if self.DEBUG:
            print('Setting up model...')
            
        ## fetch weights
        if state_dict_path:
            model = self.__fetch_weights__(model, state_dict_path)
        
        ## setup model
        self.__setup_model__(model, activation, device)
        
    def quickstart_model(self, state_dict_path:Optional[str] = None)->NoReturn:
        from ..deep_learning.dl_model_assembly import assembled_model
        
        if self.DEBUG:
            print('Quickstarting model...')        
        ## setup model
        self.setup_model(model = assembled_model['model'],
                         activation = assembled_model['activation'],
                         device = assembled_model['device'],
                         state_dict_path = state_dict_path)
    
    #%% Data Processing
    def __pre_process_data__(self)->NoReturn:
        if self.DEBUG:
            print('Pre-processing data...')
        
        # adjust image size
        self.__adjust_img_size__()
        
        # strip images
        self.__strip_imgs__()
        
        # normalize/standardize images
        order = [self.__normalize_imgs__, self.__standardize_imgs__]
        for func in order:
            if self.metadata[func.__name__.split('_')[2]]:
                func()

        # patchify images
        self.__patchify_imgs__()
        
        # reshape images
        self.__reshape_imgs__()
    
    def __go_to_sleep__(self)->NoReturn:       
        if self.DEBUG:
            print("Running checks...")
            
        ## checking
        try:
            assert self.__check_model__()
            assert self.__check_data_in__()
        except Exception as e:
            print(f'Error: {e}')
            return
        
        if self.DEBUG:
            print("Passing out...")
        # offloading data
        for key, patches in self.data_out.items():
            self.data_out[key] = self.__offload_data_to_device__(patches)
        
        if self.DEBUG:
            print('Gone to sleep...')
            
        with tqdm(total = len(self.data_out['data']), 
                  desc='Dreaming of nature...', 
                  file = sys.stdout) as pbar:
            for i, patch in enumerate(self.data_out['data']):
                # dreaming about nature
                self.data_out['data'][i] = self.__dream__(patch)
                
                # update progress bar
                pbar.update(1)
                
        if self.DEBUG:
            print('Waking up...')
        # fetching data
        for key, patches in self.data_out.items():
            self.data_out[key] = self.__fetch_data_from_device__(patches)
    
    def __dream__(self, patch:torch.TensorType)->np.ndarray:
        return self.activation(self.model(patch))
    
    #%% Data processing utils
    def __adjust_img_size__(self)->NoReturn:
        if self.DEBUG:
            print('Adjusting image size...')
            
        for key, img in self.data_in.items():
            self.data_in[key] = img[:,:-(img.shape[1] % self.metadata['patch_size'][0]), :-(img.shape[2] % self.metadata['patch_size'][1])]
    
    def __strip_imgs__(self)->NoReturn:
        if self.DEBUG:
            print('Stripping images...')
        for key, img in self.data_in.items():
            self.data_out[key] = img[0]
    
    def __patchify_imgs__(self)->NoReturn:
        if self.DEBUG:
            print('Patchifying images...')
        for key, img in self.data_out.items():
            self.data_out[key] = patchify(image = img,
                                          patch_size = self.metadata['patch_size'], 
                                          step=self.metadata['patch_size'][1])    
    
    def __unpatchify_imgs__(self)->NoReturn:
        if self.DEBUG:
            print('Unpatchifying images...')
        for key, patches in self.data_out.items():
            self.data_out[key] = unpatchify(patches = patches, 
                                            imsize = self.data_in['data'].shape)
    
    def __reshape_imgs__(self)->NoReturn:
        if self.DEBUG:
            print('Reshaping images...')
        for key, patches in self.data_out.items():
            num_tmp = int((np.floor(self.data_in[key].shape[1]/self.metadata['patch_size'][0]))) * int((np.floor(self.data_in[key].shape[2]/self.metadata['patch_size'][1])))
            self.data_out[key] = np.reshape(patches,(num_tmp, 1, self.metadata['patch_size'][0], self.metadata['patch_size'][1]))
    
    def __normalize_imgs__(self)->NoReturn:
        if self.DEBUG:
            print('Normalizing images...')
        for key, img in self.data_out.items():
            self.data_out[key] = (img - np.min(img))/(np.max(img)-np.min(img))
            
    def __standardize_imgs__(self)->NoReturn:
        if self.DEBUG:
            print('Standardizing images...')
        for key, img in self.data_out.items():
            self.data_out[key] = (img - np.mean(img))/np.std(img)
            
    #%% Helper Functions
    def __offload_data_to_device__(self, data:np.ndarray)->torch.TensorType:
        if self.DEBUG:
            print('Offloading data...')
        return torch.tensor(data, dtype = self.metadata["tensortype"]).to(self.device)
    
    def __fetch_data_from_device__(self, data:torch.TensorType)->np.ndarray:
        if self.DEBUG:
            print('Fetching data...')
        return data.cpu().detach().numpy()
    
    #%% checks
    def __check_model__(self)->bool:
        if self.DEBUG:
            print('Checking model...')
        if self.model == None:
            print('Error: Model not found. Please setup the model before proceeding.')
            return False
        return True

    def __check_data_in__(self)->bool:
        if self.DEBUG:
            print('Checking data...')
        if len(self.data_in) == 0:
            print('Error: Data not loaded. Please load the data before proceeding.')
            return False
        return True

