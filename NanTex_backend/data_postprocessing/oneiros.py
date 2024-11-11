## Dependencies
import sys
import torch
import numpy as np
import pathlib as pl
from skimage import io
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify

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
                 data_path_out:str = None,
                 mode:str = 'has_ground_truth',
                 data_type:str = 'npy',
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
        self.data_type = data_type
        
        # internal variables
        self.metadata = {}
        
        # DL variables
        self.model = None
        
        # Call from parent class
        self.__post_init__()
    
    @override
    def __load_npy__(self) -> None:
        if self.DEBUG:
            print('Loading npy data...')
            
        for key, path in self.data_paths_in.items():
            self.data_in[key] = np.load(path)
              
    @override  
    def __load_img__(self) -> None:
        if self.DEBUG:
            print('Loading image data...')
        
        for key, path in self.data_paths_in.items():
            self.data_in[key] = io.imread(path)
    
    @override
    def __setup_metadata__(self)->NoReturn:
        if self.DEBUG:
            print('Setting up metadata...')
        self.metadata.update(
            {
                "feature_static_threshodls" : {
                    "feature_0": 0.1,
                    "feature_1": 0.1,
                    "feature_2": 0.1
                    },
                "dynamic_thresholds": {
                    "auto_calculate": True,
                    "upper": 3,
                    "lower": -2,
                    "std_factor": 2
                    },
                "patch_size": (256, 256),
                "dream_memory_shape" : None,
                "patch_array_shape": None,
                "standardize": True,
                "normalize": False,
                "tensortype": torch.float32,
                "weights_only": True,
                "append_originals": True,
                "apply_static_thresholds": True,
                "apply_dynamic_thresholds": True
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
        files_tmp = pD.askFILES(query_title = "Please select the data files")
        data_paths_in.update({f"dream_{i}": files_tmp[i] for i in range(len(files_tmp))})

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
        model.load_state_dict(torch.load(state_dict_path, weights_only=self.metadata['weights_only']))
        
        return model
    
    #%% Fronend Functions
    def setup_model(self,
                    model:torch.nn.Sequential,
                    activation:torch.nn.Module,
                    device:torch.device,
                    output_channels:int,
                    state_dict_path:Optional[str] = None
                    )->NoReturn:
        if self.DEBUG:
            print('Setting up model...')
            
        ## fetch weights
        if state_dict_path:
            model = self.__fetch_weights__(model, state_dict_path)
        else:
            print('Warning: No state_dict path provided. Model will be initialized with random weights.')
            print("To load pre-trained weights, call the 'load_weights' method.")
        
        ## setup model
        self.__setup_model__(model, activation, device)
        
        ## memorize dream shape
        self.__memorize_dream_shape__(output_channels)
        
    def quickstart_model(self, state_dict_path:Optional[str] = None)->NoReturn:
        from ..deep_learning.dl_model_assembly import assembled_model, final_layer_config
        
        if self.DEBUG:
            print('Quickstarting model...')        
        ## setup model
        self.setup_model(model = assembled_model['model'],
                         activation = assembled_model['activation'],
                         device = assembled_model['device'],
                         state_dict_path = self.__check_filepath__(state_dict_path, 'state_dict'),
                         output_channels=final_layer_config['out_channels'])
        
    def load_weights(self, state_dict_path:str = None)->NoReturn:
        if self.DEBUG:
            print('Loading weights...')
        
        # Check path
        state_dict_path = self.__check_filepath__(state_dict_path, 'state_dict')
         
        try:
            ## fetch weights
            self.model = self.__fetch_weights__(self.model, state_dict_path)
            self.__setup_model__(self.model, self.activation, self.device)
        except Exception as e:
            print(f'Error: {e}')
            print("This may be due to the model not being setup.")
            print("Please call 'setup_model' and try again.")
            
    
    #%% Data Processing
    def dream(self)->NoReturn:
        if self.DEBUG:
            print("Going to bed...")
            
        # run pre-processing
        self.__pre_process_data__()
        
        # offload data
        self.__offload_data_to_device__()
        
        # run checks
        self.__run_checks__()
        
        # go to sleep
        if self.DEBUG:
            print('Passing out...')
        self.__go_to_sleep__()
        
        # reconstruct images
        if self.DEBUG:
            print('Waking up...')
        self.__post_process_data__()
    
    def __post_process_data__(self)->NoReturn:
        if self.DEBUG:
            print('Wrapping up dreams...')

        # unpatchify images
        self.__unpatchify_imgs__()

        # apply thresholds
        self.__apply_thresholds__()

        # append originals
        if self.metadata['append_originals']:
            self.__append_originals__()
    
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

        # memorize patch array shape
        self.__memorize_patch_array_shape__()

        # patchify images
        self.__patchify_imgs__()
        
        # reshape images
        self.__reshape_imgs__()
    
    def __go_to_sleep__(self)->NoReturn:                   
        with tqdm(total = len(self.data_out), 
                  desc='Dreaming of nature...', 
                  file = sys.stdout,
                  position=0) as pbar:
            for key, batch in self.data_out.items():
                
                with tqdm(total = len(batch),
                          desc=f'Currently at {key}...',
                          file = sys.stdout,
                          position=1) as subpbar:
                    
                    dream_memory:np.ndarray = np.zeros(self.metadata['dream_memory_shape'][key])
                    for i in range(batch.shape[0]):
        
                        # dreaming about nature
                        dream_memory[i,...] = self.__dream__(batch[i,None,...])
                        # progress bar update
                        subpbar.update(1)
                
                self.data_out[key] = dream_memory
                # update progress bar
                pbar.update(1)

        ## handle pbars
        pbar.color = 'green'
        subpbar.color = 'green'
        pbar.set_description('Dreaming of nature... Done!')
        subpbar.set_description(f'Currently at {key}... Done!')
        
        pbar.close()
        subpbar.close()
    
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
            self.data_out[key] = {  
                                    f"feature_{i}" : unpatchify(patches = patches[:,i,:,:].reshape(*self.metadata['patch_array_shape'][key],
                                                                                                   *self.metadata['patch_size']),
                                                                imsize = self.data_in[key].shape[1:]) 
                                    for i in range(patches.shape[1])
                                  }
    
    def __reshape_imgs__(self)->NoReturn:
        if self.DEBUG:
            print('Reshaping images...')
        for key, patches in self.data_out.items():
            self.data_out[key] = np.reshape(a = patches,
                                            newshape = (self.metadata['patch_array_shape'][key][0]*self.metadata['patch_array_shape'][key][1], 
                                                        1, 
                                                        self.metadata['patch_size'][0], 
                                                        self.metadata['patch_size'][1]
                                                        )
                                            )
    
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
            
    def __memorize_dream_shape__(self, out_channels:int)->NoReturn:
        if self.DEBUG:
            print('Memorizing dream shape...')
        self.metadata['dream_memory_shape'] = {key : (batch.shape[0], out_channels, *self.metadata['patch_size']) for key, batch in self.data_out.items()}
        
    def __memorize_patch_array_shape__(self)->NoReturn:
        if self.DEBUG:
            print('Memorizing patch array shape...')
        self.metadata['patch_array_shape'] = {key :(int((np.floor(self.data_in[key].shape[1]/self.metadata['patch_size'][0]))),
                                                    int((np.floor(self.data_in[key].shape[2]/self.metadata['patch_size'][1])))) for key in self.data_out.keys()}
        
    def __apply_static_thresholds__(self)->NoReturn:
        if self.DEBUG:
            print('Applying thresholds...')
        for key, dream in self.data_out.items():
            for feature_key, feature in dream.items():
                feature[feature < self.metadata['feature_static_threshodls'][feature_key]] = 0
                
    def __apply_dynamic_thresholds__(self)->NoReturn:
        if self.DEBUG:
            print('Applying dynamic thresholds...')
        
        for key, dream in self.data_out.items():
            for feature_key, feature in dream.items():
                tmp_dyn_th = self.__generate_dynamic_thresholds__(feature)
                feature[feature <= tmp_dyn_th[0]] = 0
                feature[feature >= tmp_dyn_th[1]] = 0
                
    def __generate_dynamic_thresholds__(self, feature:np.ndarray)->Tuple[float, float]:
        if self.DEBUG:
            print('Generating dynamic thresholds...')
            
        try:
            # get histogram
            counts, bins = np.histogram(feature.ravel(), bins=100)
            
            if not self.metadata['dynamic_thresholds']["auto_calculate"]:
                    return (bins[np.argmax(counts)] - self.metadata['dynamic_thresholds']['lower'], 
                            bins[np.argmax(counts)] + self.metadata['dynamic_thresholds']['upper'])
            
            return (bins[np.argmax(counts)] - self.metadata['dynamic_thresholds']['std_factor'], 
                    bins[np.argmax(counts)] + self.metadata['dynamic_thresholds']['std_factor'])
        
        except Exception as e:
            print(f'Error: {e}')
            return (0, 0)
        
    def __apply_thresholds__(self)->NoReturn:
        if self.DEBUG:
            print('Applying thresholds...')
        
        # apply thresholds
        order = [self.__apply_dynamic_thresholds__, self.__apply_static_thresholds__]
        for func in order:
            if self.metadata[func.__name__.split('__')[1]]:
                func()

    def __append_originals__(self)->NoReturn:
        if self.DEBUG:
            print('Appending originals...')
        for key, dream in self.data_out.items():
            self.data_out[key].update({"original": self.data_in[key][0]})
            
    #%% Helper Functions
    def __offload_data_to_device__(self)->torch.TensorType:
        if self.DEBUG:
            print('Offloading data...')
        for key, patches in self.data_out.items():
            self.data_out[key] = self.__cast_data_to_tensor__(patches)

    def __cast_data_to_tensor__(self, data:np.ndarray)->torch.TensorType:
        return torch.tensor(data, dtype = self.metadata["tensortype"]).to(self.device)
    
    def __dreamcatcher__(self, data:torch.TensorType)->np.ndarray:
        return data.cpu().detach().numpy()
    
    def __dream__(self, patch:torch.TensorType)->np.ndarray:
        if not torch.any(patch):
            return patch.cpu().detach().numpy()
        return self.activation(self.model(patch)).cpu().detach().numpy()
    
#%% Visualization
# yet to come

#%% Export
    def export(self, export_path:str, type:str)->NoReturn:
        if self.DEBUG:
            print('Exporting data...')
    pass

    def __export_npy__(self, export_path:str)->NoReturn:
        if self.DEBUG:
            print('Exporting npy data...')
        
        for key, dream in self.data_out.items():
            out = np.stack([dream[feature] for feature in dream.keys()], axis=1)
            np.save(export_path + f"\\{key}.npy", out)
        
        return True
    
    def __export_png__(self, export_path:str)->NoReturn:
        if self.DEBUG:
            print('Exporting png data...')
        
        for key, dream in self.data_out.items():
            # create dream directory
            pl.Path(export_path + f"\\{key}").mkdir(parents=True, exist_ok=True)

            for feature_key, feature in dream.items():
                io.imsave(export_path + f"\\{key}\\{feature_key}.png", feature)

#%% checks   
    def __run_checks__(self)->NoReturn:
        if self.DEBUG:
            print('Running checks...')
        try:
            assert self.__check_model__()
            assert self.__check_data_in__()
            assert self.__check_data_out__()
            assert self.__check_dream_memory__()
            assert self.__check_patch_array_shape__()
        except Exception as e:
            print(f'Error: {e}')
            return
        
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
    
    def __check_data_out__(self)->bool:
        if self.DEBUG:
            print('Checking data...')
        if len(self.data_out) == 0:
            print('Error: Data not processed. Please process the data before proceeding.')
            return False
        return True
    
    def __check_dream_memory__(self)->bool:
        if self.DEBUG:
            print('Checking dream memory...')
        if self.metadata['dream_memory_shape'] == None:
            print('Error: Dream memory not set. Please set the dream memory before proceeding.')
            return False
        return True
    
    def __check_patch_array_shape__(self)->bool:
        if self.DEBUG:
            print('Checking patch shape...')
        if self.metadata['patch_array_shape'] == None:
            print('Error: Patch size not set. Please set the patch size before proceeding.')
            return False
        return True

