## Dependencies
import os
import sys
import glob
import json
import torch
import numpy as np
from pprint import pprint
import albumentations as A

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from typing import Tuple, Union, Dict, List

## Custom Dependencies
from ..Util.pyDialogue import pyDialogue as pD
from ..Util.bit_generator_utils import initialize_generator, spawn_generator, seed_generator

class DataGenerator(Dataset):
    'Load, Augment and distribute batches for training & valitation.'
    def __init__(self, 
                 files:list, 
                 dim=(256,256), 
                 in_channels:int=1, 
                 out_channels:int=3,
                 aug_line:A.Compose = None, 
                 chunks:int = 8, 
                 normalize:bool = False, 
                 standardize:bool = False, 
                 replace_raw:bool = True,
                 replace_val:bool = False,
                 batchsize:int = 32, 
                 load_per_sample:int = 8, 
                 dtype_in:np.dtype = np.uint16,
                 dtype_out:np.dtype = np.float32,
                 dtype_masks:np.dtype = np.uint8,
                 gen_type:str = 'DXSM',
                 gen_seed:int = None,
                 is_val:bool = False
                 )->None:
        """Data generator object used in training and validation to load, augment and distribute raw and validation data in batch.

        Args:
            files (list): List of overlayed data including a single-color, multi-structure image in addition to the single ground truth images.
            dim (tuple, optional): Patch Dimension (row, column). Defaults to (256,256).
            in_channels (int, optional): Number of input channels. Defaults to 1.
            out_channels (int, optional): Number of output channels. Defaults to 3.
            aug_line (A.Compose, optional): Albumentations augmentation pipeline that data passes before being returned. Defaults to None.
            chunks (int, optional): Number of patches taken from the same dataset. Defaults to 8. <- LEGACY DEBUG PARAMETER, WILL BE DEPRECATED SOON
            normalize (bool, optional): Normalize Data y/n. Defaults to False.
            standardize (bool, optional): Standardize Data (Zero-Mean, Unit-Variance) y/n. Defaults to False.
            batchsize (int, optional): Number of patches per batch. Defaults to 32.
            load_per_sample (int, optional): Number of datasets to load per batch. Number of patches taken from the same dataset. Defaults to 8.
            dtype_in (np.dtype, optional): Data type of the input data. Defaults to np.uint16.
            dtype_out (np.dtype, optional): Data type of the output data. Defaults to np.float32.
            dtype_masks (np.dtype, optional): Data type of the masks. Defaults to np.uint8.
            gen_type (str, optional): Type of random number generator. Defaults to 'DXSM'.
            gen_seed (int, optional): Seed for the random number generator. Defaults to None.
            is_val (bool, optional): Flag if the object is used for generating validation data. Defaults to False.
        """
        
        'Initialization'
        self.__dim = dim
        self.__files = files
        self.__crop_size = dim[0]
        self.__chunks = chunks
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__transform = aug_line
        self.__normalize = normalize
        self.__standardize = standardize
        self.__BatchSize = batchsize
        self.__Load_per_Sample = load_per_sample
        self.__dtype_in = dtype_in
        self.__dtype_out = dtype_out
        self.__dtype_masks = dtype_masks
        self.__is_val = is_val
        self.__replace_raw = replace_raw
        self.__replace_val = replace_val
        
        ## Check for padding if no augmentation pipeline is submitted
        if not aug_line:
            self.__transform = None
            self.__get_padding()
            
        ## Initialize Bit Generator
        self.__gen:np.random.Generator
        if gen_seed == None:
            self.__gen = initialize_generator(gen_type)
        else:
            self.__gen = seed_generator(gen_type, gen_seed)

    def __len__(self)->int:
        """Denotes the number of files per batch

        Returns:
            int: Number of files in batch
        """
        return len(self.__files)

    def __getitem__(self, index)->np.ndarray:
        """Generate one batch of data

        Args:
            index (int): index of file in dataset to load

        Returns:
            np.ndarray: X - samples, y - ground truth
        """
        if self.__BatchSize > 1:    
            # setup the batch
            X = np.empty((self.__BatchSize, self.__in_channels, *self.__dim), dtype = self.__dtype_out)
            y = np.empty((self.__BatchSize, self.__out_channels, *self.__dim), dtype = self.__dtype_out)
                    
            if self.__is_val:            
                tmp_list = self.__gen.choice(a = self.__files, 
                                            size = int(self.__BatchSize//self.__Load_per_Sample), 
                                            replace = self.__replace_val)
            else:
                tmp_list = self.__gen.choice(a = self.__files, 
                                            size = int(self.__BatchSize//self.__Load_per_Sample), 
                                            replace = self.__replace_raw)
            
            for i, in_file in enumerate(tmp_list):
                for j in range(self.__Load_per_Sample):
                    X[i*self.__Load_per_Sample + j,...], y[i*self.__Load_per_Sample + j,...] = self.__data_generation__(file = in_file)
        
        elif self.__BatchSize == 1:
            X, y = self.__data_generation__(file = self.__files[index])

        else:
            raise ValueError("BatchSize must be greater than 0.")

        return X, y
    
    def __get_padding(self):
        # NOT NEEDED IF ALL IMAGES ARE SAME SIZE
        # if self.__is_val: 
        #     if self.__crop_size % self.__chunks == 0:
        #         self.__chunks = 4000
        #     else:
        #         self.__chunks = self.__crop_size + (self.__crop_size % self.__chunks)
                
        if not self.__is_val:
            if self.__crop_size % self.__chunks == 0:
                self.__chunks = self.__crop_size
            else:
                self.__chunks = self.__crop_size + (self.__crop_size % self.__chunks)
  
    def __data_augmentation__(self, 
                              image:np.ndarray, 
                              masks:np.ndarray
                              )->Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Perform data augmentation as built in the Albumentations library.

        Args:
            images (np.ndarray): Single-color, multi-structure image or RGB images <- not yet implemented
            masks (np.ndarray): Stack of ground truth structures

        Returns:
            dict: Augmented images and masks
        """
        tmp = self.__transform(image = np.squeeze(image).astype(self.__dtype_in), 
                               masks = list(masks.astype(self.__dtype_in)))
        
        return tmp['image'], tmp['masks']

    def __data_generation__(self, file:str)->Tuple[np.ndarray, np.ndarray]:
        """Generates data containing # batch_size augmented samples and likewise augmented ground truth.

        Returns:
            np.ndarray: X - augmentd sample stack, 
            y - augmented ground truth stack
        """
        
        # Initialization
        X = np.empty((self.__in_channels, *self.__dim), dtype = self.__dtype_out)
        y = np.empty((self.__out_channels, *self.__dim), dtype = self.__dtype_out)
                 
        tmp = np.load(file = file).astype(self.__dtype_in)
                
        # No Augmentation
        if self.__transform == None:
            self.__transform = A.Compose([A.RandomCrop(*self.__dim,
                                                       always_apply=True),
                                          A.PadIfNeeded(min_height = self.__chunks, 
                                                        min_width = self.__chunks,
                                                        p = 1,
                                                        border_mode = 0)])


        img,masks = self.__data_augmentation__(image = tmp[self.__out_channels:],
                                               masks = tmp[:self.__out_channels])
        
        # Store sample
        if self.__normalize:
            img = torch.from_numpy(img.astype(self.__dtype_out))
            img = img/(255 * self.__out_channels) ## normalize if all superimposed imags have the same intensity distribution
            X = torch.nan_to_num(img)
            
        elif self.__standardize:
            img = torch.from_numpy(img.astype(self.__dtype_out))
            img = ((img - img.mean())/img.std())
            X = torch.nan_to_num(img)
        
        elif not self.__normalize and not self.__standardize:
            X = img
        
        if self.__in_channels == 1:
            X = X[None,...]

        # Store class
        masks = np.stack(masks,axis=0)
        if self.__normalize or self.__standardize:
            y = torch.from_numpy(masks.astype(self.__dtype_masks))
            y = y/255
                        
        elif not self.__normalize and not self.__standardize:
            y = torch.from_numpy(masks.astype(self.__dtype_masks))
        
        if self.__out_channels == 1:
            y = y[None,...]

        return X,y
    
    #%% Dunder Dethods
    def get_dict(self)->Dict:
        return self.__dict__
            
#%% Covnenience Class
class BatchDataLoader_Handler():
    
    config:Dict[str, Union[str, int, float, bool]]
    data_path_container:Dict[str, List[str]]
    datatype:str
    DEBUG:bool
    
    def __init__(self, config:Dict = None, datatype:str = 'npy', DEBUG:bool = False) -> None:
        self.config = config
        self.DEBUG = DEBUG
        self.datatype = datatype
        self.data_path_container = {"raw_source":None, "val_source":None}
    
        self.__post_init__()
    
    def __post_init__(self)->None:
        self.__config_startup__()
        self.__check_datatype__()
        self.__check_config__()
        
        # Check if data sources are available
        if not self.__check_sources__():
            if self.DEBUG:
                print("Data sources are not available.")
            self.__call_sources__()
        
    @classmethod
    def from_config(cls, 
                    config_file_path:str = None, 
                    datatype:str = 'npy', 
                    DEBUG:str = False
                    )->'BatchDataLoader_Handler':
        
        # Fetch the configuration file path
        if config_file_path == None:
            config_file_path = pD.askFILE("Please provide the path to the configuration file.")
        
        # Load the configuration file
        with open(config_file_path, 'r') as f:
            config = json.load(f)
        return cls(config, datatype, DEBUG)
        
    #%% Convenience Functions
    @staticmethod
    def generate_biolerplate_config_file(outpath:str = None)->None:
        """
        Generate a configuration file for the data generator object. 
        This file can be used to store the configuration of the data generator object for later use.

        Returns:
            dict: Configuration file
        """
        
        if outpath == None:
            outpath = pD.askDIR("Please provide the path to the output directory.")
        
        # set the configuration dictionary
        config = {
            "raw_source":None,        # Required
            "val_source":None,        # Required
            "raw_aug": None,          # Required <- See Albumentations Documentation
            "val_aug": None,          # Required <- See Albumentations Documentation
            "chunks": 8,              # Number of patches taken from the same dataset. <- LEGACY DEBUG PARAMETER, WILL BE DEPRECATED SOON
            "load_per_sample": 8,     # Number of datasets to load per batch. <- play with it if you want
            "train_batchsize": 32,    # Number of patches per batch. <- play with it if you want, but PLEASE keep it a power of 2.
            "val_batchsize": 8,       # Might want to change that if you play with the validation scheme. Leave 1 if you have an image (not patch) based validation.
            "train_shuffle": True,    # LEAVE TRUE, unless you know what you are doing
            "val_shuffle": True,      # LEAVE TRUE, unless you know what you are doing
            "normalize": False,       # LEAVE FALSE, unless you know what you are doing
            "standardize": True,      # LEAVE TRUE, unless you know what you are doing
            "val_split":0.2,          # Classcially 80/20 split <- play with it if you want
            "dim":(256,256),          # Patch Dimension for training (row, column) <- influences the dimension of structural detail during learning.
            "val_dim":(256,256),      # Patch Dimension for validation (row, column) <- Set to whole image to test the performance on a production level. 
            "num_train_workers": 4,   # Number of workers <- run the detect_workers() function to get a recommendation. I hope you got more than 4.
            "num_val_workers": 2,     # Number of workers for validation <- run the detect_workers() function to get a recommendation. This can be lower than the training workers.
            "prefetch": 8,            # Number of batches to prefetch <- based on your computational resources.
            "autobatch": True,        # LEAVE TRUE, unless you know what you are doing
            "in_channels":1,          # Number of input channels <- 1 for grayscale, 3 for RGB | For the sake of NanTex, we use 1. Leave it.
            "out_channels":3,         # Number of output channels <- Number of structures to segment.
            "dtype_in":"uint16",      # Data type of the input data <- Leave it.
            "dtype_out":"float32",    # Data type of the output data <- Leave it.
            "dtype_masks":"uint8",    # Data type of the masks <- Leave it.
            "gen_type":"DXSM",        # Type of random number generator <- Leave it, if you don't know what you are doing.
            "gen_seed":None,          # Seed for the random number generator <- Use for reproducibility.
            "pin_memory":True,        # Flag to pin memory <- Leave it.
            "persistant_workers":True,# Flag to keep workers alive <- Leave it.
            "replace_raw":True,       # Flag to replace raw data <- Leave it.
            "replace_val":False       # Flag to replace validation data <- Leave it.
            }
        
        # Dump the configuration file       
        with open(f"{outpath}/BatchDataLoader_config.json", 'w') as f:
            json.dump(config, f, indent=4, sort_keys=False)
    
    #%% Data Loader Frontend
    def setup_augmentation_pipelines(self, 
                                     raw_aug:A.Compose, 
                                     val_aug:A.Compose
                                     )->None:
        if self.DEBUG:  
            print("Setting up augmentation pipelines.")
            
        self.config["raw_aug"] = raw_aug
        self.config["val_aug"] = val_aug

    def build_BatchDataLoader(self)->Tuple[DataLoader, DataLoader]:
        # run checkups
        if self.DEBUG:
            print("Running checks.")
        try:
            assert self.__check__()
        except Exception as e:
            print(e)
            return None
        
        # fetch data
        if self.DEBUG:
            print("Fetching data.")
        self.__load_sources__()
        
        if self.DEBUG:
            print("Building Data Loader.")
        return self.__build_BatchDataLoader__(**self.config | self.data_path_container)
    
    #%% Data Loader Backend
    def __build_BatchDataLoader__(self,
                                  raw_source:str, 
                                  val_source:str = None, 
                                  val_split:float=0.1, 
                                  dim=(256,256),
                                  val_dim=(256,256),
                                  in_channels=1, 
                                  out_channels=3,
                                  raw_aug:A.Compose = None,
                                  val_aug:A.Compose = None,
                                  chunks:int = 8,
                                  load_per_sample:int = 8,
                                  train_batchsize:int = 32,
                                  val_batchsize:int = 8,
                                  train_shuffle:bool = True,
                                  val_shuffle:bool = True,
                                  replace_raw:bool = True,
                                  replace_val:bool = False,
                                  normalize:bool = False,
                                  standardize:bool = False,
                                  num_train_workers:int = 6,
                                  num_val_workers:int = 2,
                                  prefetch:int = 8,
                                  pin_memory:bool = True,
                                  persistant_workers:bool = True,
                                  autobatch:bool = True,
                                  dtype_in:np.dtype = np.uint16,
                                  dtype_out:np.dtype = np.float32,
                                  dtype_masks:np.dtype = np.uint8,
                                  gen_type:str = 'DXSM',
                                  gen_seed:int = None,
                                  )->Tuple[DataLoader, DataLoader]:
        """
        Setup DataLoad objects and feed them with data. 
        These objects are used to continuously generate batches for both training and validation. 
        Multi processing with persistent workers is highly recommended.

        Args:
            raw_source (str): Path to raw data.
            val_source (str, optional): Path to validation data if applicable. Defaults to None.
            val_split (float, optional): Percentage of raw data to use as a substitution for validation. This is NOT recommended, but better than no validation when heavy augmentation is applied to the data. Defaults to 0.1.
            dim (tuple, optional): Set raw patch dimensions in px. (row,column). Defaults to (256,256).
            val_dim (tuple, optional): Set validation patch dimensions in px. (row,column). Defaults to (256,256).
            in_channels (int, optional): Number of input channels. Each channel is used for one feature only. (e.g. gray scale image -> 1 in-channel). Defaults to 1.
            out_channels (int, optional): Number of output channels. Is equal to the number of structures to separate. Defaults to 3.
            raw_aug (A.Compose, optional): Albumentations.Composed raw data augmentation pipeline. Defaults to None.
            val_aug (A.Compose, optional): Albumentations.Composed validation data augmentation pipeline. Defaults to None.
            chunks (int, optional): Number of patches taken from the same dataset. Defaults to 8.  <- LEGACY DEBUG PARAMETER, WILL BE DEPRECATED SOON
            load_per_sample (int, optional): Number of datasets to load per batch. Defaults to 8.
            train_batchsize (int, optional): Number of patches per batch. Defaults to 32.
            train_shuffle (bool, optional): Flag to shuffle datasets between batches. Recommended. Defaults to True.
            val_shuffle (bool, optional): Flag to shuffle datasets between batches. Recommended. Defaults to True.
            normalize (bool, optional): Normalize Data y/n. Defaults to False.
            standardize (bool, optional): Standardize Data (Zero-Mean, Unit-Variance) y/n. Defaults to False.
            num_train_workers (int, optional): Number of workers for training. Defaults to 6.
            num_val_workers (int, optional): Number of workers for validation. Defaults to 2.
            prefetch (int, optional): Set how many batches should be queued per worker. Defaults to 8.
            pin_memory (bool, optional): Flag to pin memory. Defaults to True.
            persistant_workers (bool, optional): Flag to keep workers alive. Defaults to True.
            autobatch (bool, optional): Flag if you want to use the autobatching. Might increase performance. Defaults to True.
            dtype_in (np.dtype, optional): Data type of the input data. Defaults to np.uint16.
            dtype_out (np.dtype, optional): Data type of the output data. Defaults to np.float32.
            dtype_masks (np.dtype, optional): Data type of the masks. Defaults to np.uint8.
            gen_type (str, optional): Type of random number generator. Defaults to 'DXSM'.
            gen_seed (int, optional): Seed for the random number generator. Defaults to None.
            val_batchsize (int, optional): Set the number of batches used in validation. Defaults to 1.

        Returns:
            DataLoader: _description_
        """
        
        ## Initialize Random Number Generator
        tmp_gen:np.random.Generator
        if gen_seed == None:
            tmp_gen = initialize_generator(gen_type)
        else:
            tmp_gen = seed_generator(gen_type, gen_seed)
        
        if self.DEBUG:
            print("Building Data Loader.")
        
        if val_source == None:
            if (val_split < 1.0) and (val_split > 0.0):
                val_source = tmp_gen.choice(raw_source, size = int(len(raw_source)*val_split), replace = False) #test_source
                raw_source = [x for x in raw_source if x not in val_source]
        
        ## Retrun DataGenerator objects for train and test data
        
        if autobatch:
            if self.DEBUG:
                print("Using Autobatching.")
                
            train_dataset = DataGenerator(files = raw_source, 
                                          dim = dim, 
                                          in_channels = in_channels, 
                                          out_channels = out_channels, 
                                          aug_line = raw_aug, 
                                          chunks = chunks, 
                                          normalize = normalize, 
                                          standardize = standardize,
                                          batchsize = 1,
                                          load_per_sample = load_per_sample,
                                          dtype_in = dtype_in,
                                          dtype_out = dtype_out,
                                          dtype_masks = dtype_masks,
                                          gen_type = gen_type,
                                          gen_seed = gen_seed,
                                          is_val = False,
                                          replace_raw=replace_raw,
                                          replace_val=replace_val)
            
            val_dataset = DataGenerator(files = val_source, 
                                        dim = val_dim, 
                                        in_channels = in_channels,
                                        out_channels = out_channels,
                                        aug_line = val_aug,
                                        chunks = chunks,
                                        normalize = normalize, 
                                        standardize = standardize, 
                                        batchsize = 1,
                                        load_per_sample = load_per_sample,
                                        dtype_in = dtype_in,
                                        dtype_out = dtype_out,
                                        dtype_masks = dtype_masks,
                                        gen_type = gen_type,
                                        gen_seed = gen_seed,
                                        is_val = True,
                                        replace_raw=replace_raw,
                                        replace_val=replace_val)
            
            if self.DEBUG:
                print("Autobatch Data Generators created.")
                        
            return (
                DataLoader(dataset = train_dataset,
                           batch_size = train_batchsize,
                           shuffle = train_shuffle,
                           pin_memory = pin_memory,
                           num_workers = num_train_workers,
                           prefetch_factor = prefetch,
                           persistent_workers = persistant_workers),
                
                DataLoader(dataset = val_dataset,
                           batch_size = val_batchsize,
                           shuffle = val_shuffle,
                           pin_memory = pin_memory,
                           num_workers = num_val_workers,
                           prefetch_factor = prefetch,
                           persistent_workers = persistant_workers),
                )
    
        ## Retrun DataGenerator objects for train and test data
        # This is a version using the autobatching feature of the DataLoader object.
        if not autobatch:
            if self.DEBUG:
                print("Not using Autobatching.")
            
            train_dataset = DataGenerator(files = raw_source, 
                                          dim = dim, 
                                          in_channels = in_channels, 
                                          out_channels = out_channels, 
                                          aug_line = raw_aug, 
                                          chunks = chunks, 
                                          normalize = normalize, 
                                          standardize = standardize,
                                          batchsize = train_batchsize,
                                          load_per_sample = load_per_sample,
                                          dtype_in = dtype_in,
                                          dtype_out = dtype_out,
                                          dtype_masks = dtype_masks,
                                          gen_type = gen_type,
                                          gen_seed = gen_seed,
                                          is_val = False,
                                          replace_raw=replace_raw,
                                          replace_val=replace_val)
            
            val_dataset = DataGenerator(files = val_source, 
                                        dim = val_dim, 
                                        in_channels = in_channels,
                                        out_channels = out_channels,
                                        aug_line = val_aug,
                                        chunks = chunks,
                                        normalize = normalize, 
                                        standardize = standardize, 
                                        batchsize = val_batchsize,
                                        load_per_sample = load_per_sample,
                                        dtype_in = dtype_in,
                                        dtype_out = dtype_out,
                                        dtype_masks = dtype_masks,
                                        gen_type = gen_type,
                                        gen_seed = gen_seed,
                                        is_val = True,
                                        replace_raw=replace_raw,
                                        replace_val=replace_val)
            
            if self.DEBUG:
                print("Non-Autobatched Data Generators created.")
            
            return (
                DataLoader(dataset = train_dataset,
                           batch_size = None,
                           shuffle = train_shuffle,
                           pin_memory = pin_memory,
                           num_workers = num_train_workers,
                           prefetch_factor = prefetch,
                           persistent_workers = persistant_workers),
                
                DataLoader(dataset = val_dataset,
                           batch_size = None,
                           shuffle = val_shuffle,
                           pin_memory = pin_memory,
                           num_workers = num_val_workers,
                           prefetch_factor = prefetch,
                           persistent_workers = persistant_workers),
                )

    #%% Helper Functions
    def __load_config__(self, config_path:str)->None:
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def __load_sources__(self)->None:
        if self.DEBUG:
            print("Loading data sources.")
        
        self.__load_raw_source__()
        self.__load_val_source__()
    
    def __call_sources__(self)->None:
        if self.DEBUG:
            print("Loading data sources.")
        
        self.__call_raw_source__()
        self.__call_val_source__()
        
    def __call_raw_source__(self)->None:
        if self.DEBUG:
            print("Callling raw data source.")
        
        self.config["raw_source"] = pD.askDIR("Please provide the path to the raw data source.")
    
    def __call_val_source__(self)->None:
        if self.DEBUG:
            print("Callling validation data source.")
        
        self.config["val_source"] = pD.askDIR("Please provide the path to the validation data source.")
    
    def __load_raw_source__(self)->None:
        if self.DEBUG:
            print("Loading raw data source.")
        
        self.data_path_container["raw_source"] = glob.glob(f"{self.config['raw_source']}/*.{self.datatype}")
        
    def __load_val_source__(self)->None:
        if self.DEBUG:
            print("Loading validation data source.")
        
        self.data_path_container["val_source"] = glob.glob(f"{self.config['val_source']}/*.{self.datatype}")
        
    def get_config(self)->Dict:
        return self.config
    
    def pprint_config(self)->None:
        pprint(self.config)
    
    def print_config(self)->None:
        print(json.dumps(self.config, indent=4, sort_keys=False))
        
    def load_config(self, config_path:str = None)->None:
        if config_path == None:
            self.__load_config__(pD.askFILE("Please provide the path to the configuration file."))
        self.__load_config__(config_path)
        
    def call_sources(self)->None:
        self.__call_sources__()
    
    def load_sources(self)->None:
        self.__load_sources__()

    #%% Checkups
    def __config_startup__(self)->None:
        if self.DEBUG:
            print("Checking for configuration startup.")
        
        # Ceck for the configuration file
        if self.config == None:
            if not self.__check_config__():
                self.generate_biolerplate_config_file(outpath = f"{os.getcwd()}/config")
            self.__load_config__(config_path=f"{os.getcwd()}/BatchDataLoader_config.json")
            
    def __check__(self)->bool:
        return all([self.__check_config__(), self.__check_sources__(), self.__check_augmentations__(), self.__check_datatype__()])

    def __check_config__(self)->bool:
        if self.DEBUG:
            print("Checking configuration.")
        
        try:
            assert self.config != None
            assert type(self.config) == dict
            return True
        except Exception as e:
            print(e)
            return False
            
    def __check_sources__(self)->bool:
        if self.DEBUG:
            print("Checking data sources.")
        
        try:
            assert all([key in self.config.keys() for key in ["raw_source", "val_source"]])
            assert type(self.config["raw_source"]) == str
            assert type(self.config["val_source"]) == str
            return True
        except Exception as e:
            print(e)
            return False
        
    def __check_datatype__(self)->bool:
        if self.DEBUG:
            print("Checking datatype.")
        
        try:
            assert self.datatype in ["npy", "tif", "tiff", "png", "jpg", "jpeg", "bmp"]
            return True
        except Exception as e:
            print("Datatype not supported.")
            print(e)
            return False
    
    def __check_augmentations__(self)->bool:
        if self.DEBUG:
            print("Checking augmentations.")
            
        try:
            assert all([key in self.config.keys() for key in ["raw_aug", "val_aug"]])
            assert type(self.config["raw_aug"]) == A.Compose
            assert type(self.config["val_aug"]) == A.Compose
            return True
        except Exception as e:
            print(e)
            print("Use setup_augmentation_pipelines() function to set up the augmentation pipelines.")
            return False