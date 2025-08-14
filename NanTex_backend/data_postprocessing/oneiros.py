## Dependencies
import os
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
    data_dream:Dict[str, np.ndarray]
    
    DEBUG:bool
    mode:str
    metadata:Dict[str, Any]
    shape_adjuster:int
    
    model:Optional[torch.nn.Module]
    num_features:int
    num_channels_out:int
    
    def __init__(self,
                 num_features:int,
                 data_paths_in:Dict[str, List[str]],
                 data_path_out:str = None,
                 num_channels_out:int = None,
                 mode:str = 'has_ground_truth',
                 DEBUG:bool = False
                 ) -> None:
        # data variables
        self.data_paths_in = data_paths_in
        self.data_path_out = data_path_out
        self.data_in = {}  
        self.data_dream = {}
        self.data_out = {}
        
        # model variables
        self.num_features = num_features
        if not num_channels_out:
            num_channels_out = num_features
        self.num_channels_out = num_channels_out

        # control variables
        self.DEBUG = DEBUG
        self.mode = mode
        
        # internal variables
        self.metadata = {}
        self.shape_adjuster = 0
        
        # DL variables
        self.model = None
        
        # Call from parent class
        self.__post_init__()
    
    @override
    def __post_init__(self)->NoReturn:
        try:
            self.__load_data__()
            self.__setup_metadata__()
            
            if self.mode == 'has_ground_truth':
                self.shape_adjuster = 1
            
            if self.DEBUG:
                print('Oneiros Initialized...')
            
        except Exception as e:
            print(f'Error: {e}')
    
    @override
    def __setup_metadata__(self)->NoReturn:
        if self.DEBUG:
            print('Setting up metadata...')
        self.metadata.update(
            {
                "feature_static_thresholds" : {
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
                "out_type": np.uint8,
                "weights_only": True,
                "cast_all_to_img": True,
                "append_original_features": True,
                "append_original_overlays": True,
                "append_dream_overlays": True,
                "apply_static_thresholds": True,
                "apply_dynamic_thresholds": True
            }
        )
    
    #%% Classmethods Windows
    @classmethod
    def from_explorer(cls, **kwargs)->'Oneiros':
        ## initialize
        data_paths_in:Dict[str, List[str]] = {}
        
        ## check mode
        try:
            assert isinstance(kwargs.get('mode'), str)
            assert kwargs.get('mode') in ['has_ground_truth', 'no_ground_truth']
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
                   **kwargs)
        
    @classmethod
    def with_ground_truth(cls, **kwargs)->'Oneiros':
        return cls.from_explorer(mode = 'has_ground_truth', **kwargs)
    
    @classmethod
    def without_ground_truth(cls, **kwargs)->'Oneiros':
        return cls.from_explorer(mode = 'no_ground_truth', **kwargs)
    
    #%% Classmethods Linux
    @classmethod
    def from_glob(cls, *args, **kwargs)->'Oneiros':
        ## initialize
        data_paths_in:Dict[str, List[str]] = {}
        
        ## check mode
        try:
            assert isinstance(kwargs.get('mode'), str)
            assert kwargs.get('mode') in ['has_ground_truth', 'no_ground_truth']
        except Exception as e:
            print(f'Error: {e}')
            print('Please select a valid mode from the following:')
            print('1. has_ground_truth')
            print('2. no_ground_truth')
            return None
        
        # construct dream_dict
        for i, arg in enumerate(args):
            if not os.path.isfile(arg):
                raise ValueError(f"Error: Argument {i} is not a valid file.")
            data_paths_in.update({f"dream_{i}": arg})
        
        return cls(data_paths_in = data_paths_in,
                   **kwargs)
        
    
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
        
    def jumpstart_model(self, state_dict_path:Optional[str] = None)->NoReturn:
        from ..deep_learning.dl_model_assembly import assembled_model, final_layer_config
        
        if self.DEBUG:
            print('Quickstarting model...')        
        ## setup model
        self.setup_model(model = assembled_model['model'],
                         activation = assembled_model['activation'],
                         device = assembled_model['device'],
                         state_dict_path = self.__check_filepath__(state_dict_path, 'state_dict'))
        
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
        # run pre-processing
        if self.DEBUG:
            print("Going to bed...")
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
    
    def __pre_process_data__(self)->NoReturn:
        if self.DEBUG:
            print('Pre-processing data...')
            
        # check for and apply channel padding if needed 
        # to match ground truth and number of feature channels
        self.__pad_img_channels__()
        
        # adjust image size
        self.__adjust_img_size__()
        
        # strip images
        self.__strip_imgs__()
        
        # normalize/standardize images
        self.__apply_transforms__()

        # memorize patch array shape
        self.__memorize_patch_array_shape__()

        # patchify images
        self.__patchify_imgs__()
        
        # reshape images
        self.__reshape_imgs__()
        
        # memorize dream shape
        self.__memorize_dream_shape__()
        
    def __post_process_data__(self)->NoReturn:
        if self.DEBUG:
            print('Wrapping up dreams...')

        # unpatchify images
        self.__unpatchify_imgs__()

        # apply thresholds
        self.__apply_thresholds__()
        
        # append originals
        self.__append_to_output__()
    
        # cast to image
        if self.metadata['cast_all_to_img']:
            self.__cast_all_to_img__()
    
    def __go_to_sleep__(self)->NoReturn:                   
        with tqdm(total = len(self.data_dream), 
                  desc='Dreaming of nature...', 
                  file = sys.stdout,
                  position=0) as pbar:
            for key, batch in self.data_dream.items():
                
                with tqdm(total = len(batch),
                          desc=f'{key}...'.capitalize(),
                          file = sys.stdout,
                          position=1) as subpbar:
                    
                    dream_memory:np.ndarray = np.zeros(self.metadata['dream_memory_shape'][key])
                    for i in range(batch.shape[0]):
        
                        # dreaming about nature
                        dream_memory[i,...] = self.__dream__(batch[i,None,...])
                        # progress bar update
                        subpbar.update(1)
                
                self.data_dream[key] = dream_memory
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
    def __pad_img_channels__(self)->NoReturn:
        match (self.__check_has_ground_truth__(), self.__check_needs_channel_padding__()):
            case (True, True):
                if self.DEBUG:
                    print("Mode: 'has_ground_truth'")
                    print('Mismatch between the number of ground truth images and feature channels detected.')
                    print("Applying zero padding to match ground truth and feature channels.")
                    print('Continuing with padded channels...')
                
                # pad channels
                for key, img in self.data_in.items():
                    pad = np.zeros((self.num_features - (img.shape[0] - 1), # -1 for overlay
                                    img.shape[1], 
                                    img.shape[2])) 
                    self.data_in[key] = np.insert(img, -1, pad, axis=0)
                return
            
            case (True, False):
                if self.DEBUG:
                    print("Mode: 'has_ground_truth'")
                    print('No mismatch between the number of ground truth images and feature channels detected.')
                    print('Continuing with provided channels...')
                    return
                
            case _:
                if self.DEBUG:
                    print("Something went wrong...")
                    print("Please check the data and try again.")
                return
    
    def __adjust_img_size__(self)->NoReturn:
        if self.DEBUG:
            print('Adjusting image size...')
        for key, img in self.data_in.items():
            if (img.shape[0 + self.shape_adjuster] % self.metadata['patch_size'][0] != 0) or (img.shape[1 + self.shape_adjuster] % self.metadata['patch_size'][1] != 0):
                self.data_in[key] = img[:,:-(img.shape[0 + self.shape_adjuster] % self.metadata['patch_size'][0]), :-(img.shape[1 + self.shape_adjuster] % self.metadata['patch_size'][1])]
            else:
                self.data_in[key] = img
    
    def __strip_imgs__(self)->NoReturn:
        if self.DEBUG:
            print('Stripping images...')
            
        if self.mode == 'has_ground_truth':
            for key, img in self.data_in.items():
                self.data_dream[key] = img[-1]
                
        elif self.mode == 'no_ground_truth':
            for key, img in self.data_in.items():
                self.data_dream[key] = img
    
    def __patchify_imgs__(self)->NoReturn:
        if self.DEBUG:
            print('Patchifying images...')
        for key, img in self.data_dream.items():
            self.data_dream[key] = patchify(image = img,
                                            patch_size = self.metadata['patch_size'], 
                                            step=self.metadata['patch_size'][1])    
    
    def __unpatchify_imgs__(self)->NoReturn:
        if self.DEBUG:
            print('Unpatchifying images...')
        for key, patches in self.data_dream.items():
            self.data_out[key] = {  
                                    f"feature_{i}" : unpatchify(patches = patches[:,i,:,:].reshape(*self.metadata['patch_array_shape'][key],
                                                                                                   *self.metadata['patch_size']),
                                                                imsize = self.data_in[key].shape[0 + self.shape_adjuster:]) 
                                    for i in range(patches.shape[1])
                                  }
    
    def __reshape_imgs__(self)->NoReturn:
        if self.DEBUG:
            print('Reshaping images...')
        for key, patches in self.data_dream.items():
            self.data_dream[key] = np.reshape(a = patches,
                                            newshape = (self.metadata['patch_array_shape'][key][0]*self.metadata['patch_array_shape'][key][1], 
                                                        1, 
                                                        self.metadata['patch_size'][0], 
                                                        self.metadata['patch_size'][1]
                                                        )
                                            )
    
    def __normalize_imgs__(self)->NoReturn:
        if self.DEBUG:
            print('Normalizing images...')
        for key, img in self.data_dream.items():
            self.data_dream[key] = (img - np.min(img))/(np.max(img)-np.min(img))
            
    def __standardize_imgs__(self)->NoReturn:
        if self.DEBUG:
            print('Standardizing images...')
        for key, img in self.data_dream.items():
            self.data_dream[key] = (img - np.mean(img))/np.std(img)
            
    def __apply_transforms__(self)->NoReturn:
        order = [self.__normalize_imgs__, self.__standardize_imgs__]
        for func in order:
            if self.metadata[func.__name__.split('_')[2]]:
                func()
            
    def __memorize_dream_shape__(self)->NoReturn:
        if self.DEBUG:
            print('Memorizing dream shape...')
        self.metadata['dream_memory_shape'] = {key : (batch.shape[0], self.num_channels_out, *self.metadata['patch_size']) for key, batch in self.data_dream.items()}
        
    def __memorize_patch_array_shape__(self)->NoReturn:
        if self.DEBUG:
            print('Memorizing patch array shape...')
        self.metadata['patch_array_shape'] = {key :(int((np.floor(self.data_in[key].shape[0 + self.shape_adjuster]/self.metadata['patch_size'][0]))),
                                                    int((np.floor(self.data_in[key].shape[1 + self.shape_adjuster]/self.metadata['patch_size'][1])))) for key in self.data_dream.keys()}
        
    def __apply_static_thresholds__(self)->NoReturn:
        if self.DEBUG:
            print('Applying thresholds...')
        for key, dream in self.data_out.items():
            for feature_key, feature in dream.items():
                # skip overlays
                if feature_key in ['original_overlay', 'dream_overlay'] + [f"original_feature_{i}" for i in range(0, self.num_features)]:
                    continue
                # apply thresholds
                feature[feature < self.metadata['feature_static_thresholds'][feature_key]] = 0
                
    def __apply_dynamic_thresholds__(self)->NoReturn:
        if self.DEBUG:
            print('Applying dynamic thresholds...')
        
        for key, dream in self.data_out.items():
            for feature_key, feature in dream.items():
                # skip overlays
                if feature_key in ['original_overlay', 'dream_overlay'] + [f"original_feature_{i}" for i in range(0, self.num_features)]:
                    continue
                
                # apply thresholds
                tmp_dyn_th = self.__generate_dynamic_thresholds__(feature)
                threshold_array = (feature >= tmp_dyn_th[0]) * (feature <= tmp_dyn_th[1])
                feature[threshold_array] = 0
                
                
    def __generate_dynamic_thresholds__(self, feature:np.ndarray)->Tuple[float, float]:
        if self.DEBUG:
            print('Generating dynamic thresholds...')
            
        try:
            # get histogram
            counts, bins = np.histogram(feature.ravel(), bins=100)
            
            if not self.metadata['dynamic_thresholds']["auto_calculate"]:
                    return (bins[np.argmax(counts) + self.metadata['dynamic_thresholds']['lower']], 
                            bins[np.argmax(counts) + self.metadata['dynamic_thresholds']['upper']])
            
            return (bins[np.argmax(counts)] - self.metadata['dynamic_thresholds']['std_factor'] * np.std(feature.ravel()), 
                    bins[np.argmax(counts)] + self.metadata['dynamic_thresholds']['std_factor'] * np.std(feature.ravel()))
        
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
                
    def __cast_all_to_img__(self)->NoReturn:
        if self.DEBUG:
            print('Casting to image...')
        for key, dream in self.data_out.items():
            for feature_key, feature in dream.items():
                self.data_out[key][feature_key] = self.__cast_to_img__(feature)

    def __append_original_overlays__(self)->NoReturn:
        if self.DEBUG:
            print('Appending original overlays...')
        
        if self.mode == 'has_ground_truth':
            for key, dream in self.data_out.items():
                self.data_out[key].update({"original_overlay": self.data_in[key][-1]})
        
        elif self.mode == 'no_ground_truth':
            for key, dream in self.data_out.items():
                self.data_out[key].update({"original_overlay": self.data_in[key]})
            
    def __append_original_features__(self)->NoReturn:
        if self.DEBUG:
            print('Appending original features...')
        
        # fail catch
        if self.mode == 'has_ground_truth':
            for key, dream in self.data_out.items():
                for i in range(0, self.num_features):
                    self.data_out[key].update({f"original_feature_{i}": self.data_in[key][i]})
            
    def __append_dream_overlays__(self)->NoReturn:
        if self.DEBUG:
            print('Appending dreams...')
        for key, dream in self.data_out.items():
            # construct dream overlay
            out:np.ndarray = np.sum([dream[f"feature_{i}"] for i in range(self.num_channels_out)], axis=0)
            out = self.__cast_to_img__(out)
            self.data_out[key].update({"dream_overlay": out})
            
    def __append_to_output__(self)->NoReturn:
        if self.DEBUG:
            print('Appending data...')
        
        order = [self.__append_original_overlays__, self.__append_original_features__, self.__append_dream_overlays__]
        for func in order:
            if self.metadata[func.__name__.split('__')[1]]:
                func()
            
    #%% Helper Functions
    def __offload_data_to_device__(self)->torch.TensorType:
        if self.DEBUG:
            print('Offloading data...')
        for key, patches in self.data_dream.items():
            self.data_dream[key] = self.__cast_data_to_tensor__(patches)

    def __cast_data_to_tensor__(self, data:np.ndarray)->torch.TensorType:
        return torch.tensor(data, dtype = self.metadata["tensortype"]).to(self.device)
    
    def __dreamcatcher__(self, data:torch.TensorType)->np.ndarray:
        return data.cpu().detach().numpy()
    
    def __dream__(self, patch:torch.TensorType)->np.ndarray:
        if not torch.any(patch):
            return patch.cpu().detach().numpy()
        return self.activation(self.model(patch)).cpu().detach().numpy()
    
    def __normalize__(self, data:np.ndarray)->np.ndarray:
        return (data - np.min(data))/(np.max(data)-np.min(data))
    
    def __cast_to_img__(self, data:np.ndarray)->np.ndarray:
        if not np.any(data): # catch for empty arrays
            return data.astype(self.metadata['out_type'])
        return (self.__normalize__(data) * np.iinfo(self.metadata['out_type']).max).astype(self.metadata['out_type'])
    
    def __num_panels__(self)->int:
        return self.num_channels_out + 1
    
    def __num_dreams__(self)->int:
        return len(self.data_in.keys())
    
#%% Convenience Functions
    def change_mode(self, mode:str)->NoReturn:
        if self.DEBUG:
            print('Changing mode...')
        self.mode = mode
    
    def change_num_features(self, num_features:int)->NoReturn:
        if self.DEBUG:
            print('Changing number of features...')
        self.num_features = num_features
    
    def update_metadata(self, metadata:Dict[str, Any])->NoReturn:
        if self.DEBUG:
            print('Updating metadata...')
        self.metadata.update(metadata)
    
    def change_input_data(self, data_paths_in:Dict[str, List[str]])->NoReturn:
        if self.DEBUG:
            print('Changing input data...')
        self.data_paths_in = data_paths_in
    
    def load_new_data_windows(self)->NoReturn:
        if self.DEBUG:
            print('Calling new input data...')
        # get data paths
        files_tmp = pD.askFILES(query_title = "Please select the data files")
        self.data_paths_in = {f"dream_{i}": files_tmp[i] for i in range(len(files_tmp))}
        
        # reset
        self.reset_data()
        
        # load data
        self.__load_data__()
        
        # return
        print('New data retrieved...')
        print('Please call the dream method to process the data.')
        
    def load_new_data_linux(self, *args)->NoReturn:
        if self.DEBUG:
            print('Calling new input data...')
        # construct dream_dict
        for i, arg in enumerate(args):
            if not os.path.isfile(arg):
                raise ValueError(f"Error: Argument {i} is not a valid file.")
            self.data_paths_in.update({f"dream_{i}": arg})
        
        # reset
        self.reset_data()
        
        # load data
        self.__load_data__()
        
        # return
        print('New data retrieved...')
        print('Please call the dream method to process the data.')
    
    def set_DEBUG(self, DEBUG:bool)->NoReturn:
        if self.DEBUG:
            print('Changing verbosity...')
        self.DEBUG = DEBUG
        
    def reset_data(self)->NoReturn:
        if self.DEBUG:
            print('Resetting Oneiros...')
        self.data_in = {}
        self.data_dream = {}
        self.data_out = {}
    
#%% Visualization
    def visualize(self,
                  dream_no:int,
                  cmap:str = 'gray',
                  ticks:bool = False,
                  return_fig_axs:bool = True
                  )->Union[Tuple[plt.Figure, plt.Axes], NoReturn]:
        
        if self.DEBUG:
            print('Visualizing data...')
            
        # run checks
        self.__check_is_data_out__()
        
        # check mode
        match self.mode:
            case 'has_ground_truth':
                return self.__visualize_with_ground_truth__(dream_no = dream_no,
                                                            cmap = cmap,
                                                            ticks = ticks,
                                                            return_fig_axs = return_fig_axs)	
            case 'no_ground_truth':
                return self.__visualize_without_ground_truth__(dream_no = dream_no,
                                                               cmap = cmap,
                                                               ticks = ticks,
                                                               return_fig_axs = return_fig_axs)
            case _:
                print('Error: Mode not supported yet...')

    def __visualize_with_ground_truth__(self,
                                        dream_no:int,
                                        cmap:str,
                                        ticks:bool,
                                        return_fig_axs:bool
                                        )->Union[Tuple[plt.Figure, plt.Axes], NoReturn]:
        
        fig, axs = plt.subplots(2, self.__num_panels__(), figsize=(5*self.__num_panels__(), 10), dpi = 150)
        plt.subplots_adjust(wspace=0.2, hspace=0.2)
        
        # append dream overlay and original
        if not self.metadata['append_dream_overlays']:
            self.__append_dream_overlays__()
        if not self.metadata['append_original_overlays']:
            self.__append_original_overlays__()

        # plot
        for i in range(0, self.num_features): # we iterate over all present features
            axs[0,i].imshow(self.__cast_to_img__(self.data_in[f"dream_{dream_no}"][i]), cmap=cmap) 
            axs[0,i].set_title(f"Ground Truth {i}")

        for i in range(0, self.num_channels_out): # we iterate over all present predicitions
            axs[1,i].imshow(self.__cast_to_img__(self.data_out[f"dream_{dream_no}"][f"feature_{i}"]), cmap=cmap)
            axs[1,i].set_title(f"Feature Dream {i}")
                        
        # add overlays
        axs[0,-1].imshow(self.__cast_to_img__(self.data_out[f"dream_{dream_no}"]["original_overlay"]), cmap=cmap)
        axs[1,-1].imshow(self.__cast_to_img__(self.data_out[f"dream_{dream_no}"]["dream_overlay"]), cmap=cmap)
        
        axs[0,-1].set_title(f"Original Overlay")
        axs[1,-1].set_title(f"Dream Overlay")
        
        # layout
        if not ticks:
            for ax in axs.ravel():
                ax.tick_params(axis='both', 
                               which='both', 
                               bottom=False, 
                               top=False, 
                               left=False, 
                               right=False, 
                               labelbottom=False, 
                               labelleft=False)
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            
            
        if return_fig_axs:
            return fig, axs
        
    def __visualize_without_ground_truth__(self,
                                           dream_no:int,
                                           cmap:str,
                                           ticks:bool,
                                           return_fig_axs:bool
                                           )->Union[Tuple[plt.Figure, plt.Axes], NoReturn]:
        
        fig, axs = plt.subplots(1, self.__num_panels__() + 1, figsize=(5*(self.__num_panels__()) + 1, 5), dpi = 150)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        
        # append dream overlay and original
        if not self.metadata['append_dream_overlays']:
            self.__append_dream_overlays__()
        if not self.metadata['append_original_overlays']:
            self.__append_original_overlays__()
        
        # plot
        for i in range(0, self.num_channels_out):
            axs[i + 1].imshow(self.__cast_to_img__(self.data_out[f"dream_{dream_no}"][f"feature_{i}"]), cmap=cmap)
            axs[i + 1].set_title(f"Feature Dream {i}")
                        
        # add overlays
        axs[0].imshow(self.__cast_to_img__(self.data_out[f"dream_{dream_no}"]["original_overlay"]), cmap=cmap)
        axs[-1].imshow(self.__cast_to_img__(self.data_out[f"dream_{dream_no}"]["dream_overlay"]), cmap=cmap)
        
        axs[0].set_title(f"Original Overlay")
        axs[-1].set_title(f"Dream Overlay")
        
        # layout
        if not ticks:
            for ax in axs.ravel():
                ax.tick_params(axis='both', 
                               which='both', 
                               bottom=False, 
                               top=False, 
                               left=False, 
                               right=False, 
                               labelbottom=False, 
                               labelleft=False)

        if return_fig_axs:
            return fig, axs

#%% Export
    def export(self, out_type:str, outpath:str = None)->NoReturn:
        if self.DEBUG:
            print('Exporting data...')    
            
        # check outpath
        if outpath:
            self.data_path_out = outpath
        outpath = self.__check_outpath__()
        
        # match out_type
        match out_type:
            case 'npy':
                self.__export_npy__(outpath)
            case 'png':
                self.__export_png__(outpath)
            case 'single_npy':
                self.__export_stacked_npy__(outpath)
            case _:
                print('Error: Data type not supported yet...')

    def __export_stacked_npy__(self, export_path:str)->NoReturn:
        if self.DEBUG:
            print('Exporting to stacked npy ...')
        
        for key, dream in self.data_out.items():
            out = np.stack([dream[feature] for feature in dream.keys()], axis=0)
            np.save(export_path + f"\\{key}.npy", out)
            
    def __export_npy__(self, export_path:str)->NoReturn:
        if self.DEBUG:
            print('Exporting npy data...')
        
        for key, dream in self.data_out.items():
            # create dream directory
            pl.Path(export_path + f"\\{key}").mkdir(parents=True, exist_ok=True)

            for feature_key, feature in dream.items():
                np.save(export_path + f"\\{key}\\{feature_key}.npy", feature)
            
        
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
            assert self.__check_data_dream__()
            assert self.__check_dream_memory__()
            assert self.__check_patch_array_shape__()
            assert self.__check_num_features__()
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
    
    def __check_data_dream__(self)->bool:
        if self.DEBUG:
            print('Checking data...')
        if len(self.data_dream) == 0:
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
    
    def __check_is_data_dream__(self)->bool:
        if self.DEBUG:
            print('Checking data...')
        if len(self.data_dream) == 0:
            print('Error: Data not processed. Please process the data before proceeding.')
            return False
        return True
    
    def __check_is_data_out__(self)->bool:
        if self.DEBUG:
            print('Checking data...')
        if len(self.data_out) == 0:
            print('Error: Data not processed. Please process the data before proceeding.')
            return False
        return True
    
    def __check_num_features__(self)->int:
        if self.mode == 'has_ground_truth':
            if self.DEBUG:
                print('Checking number of features...')
            try:
                for key, dream in self.data_out.items():
                    assert (len(dream) - 1) == self.num_features
            except Exception as e:
                print(f'Error: {e}')
                return False
        return True

    ## Case specific checks
    def __check_has_ground_truth__(self)->bool:
        if self.DEBUG:
            print('Checking ground truth...')
        if self.mode == 'has_ground_truth':
            return True
        return False
    
    def __check_needs_channel_padding__(self)->bool:
        if self.DEBUG:
            print('Checking channel padding...')
        for key, img in self.data_in.items():
            if img.shape[1] != self.num_features:
                return True
        return False
