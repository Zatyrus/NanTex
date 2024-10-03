## Dependencies
import sys, json
import torch
import numpy as np
import albumentations as A

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

## Custom Dependencies
from ..Util.pyDialogue import pyDialogue as pD

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
                 batchsize:int = 32, 
                 load_per_sample:int = 8, 
                 is_val:bool = False
                 )->None:
        """Data generator object used in training and validation to load, augment and distribute raw and validation data in batch.

        Args:
            files (list): List of overlayed data including a single-color, multi-structure image in addition to the single ground truth images.
            dim (tuple, optional): Patch Dimension (row, column). Defaults to (256,256).
            in_channels (int, optional): Number of input channels. Defaults to 1.
            out_channels (int, optional): Number of output channels. Defaults to 3.
            aug_line (A.Compose, optional): Albumentations augmentation pipeline that data passes before being returned. Defaults to None.
            chunks (int, optional): Number of patches taken from the same dataset. Defaults to 8.
            normalize (bool, optional): Normalize Data y/n. Defaults to False.
            standardize (bool, optional): Standardize Data (Zero-Mean, Unit-Variance) y/n. Defaults to False.
            batchsize (int, optional): Number of patches per batch. Defaults to 32.
            load_per_sample (int, optional): Number of datasets to load per batch. Defaults to 8.
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
        self.__is_val = is_val
                
        self.__get_padding()

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
        if self.__BatchSize != 1:
            self._open = self.__files[index]
            X = np.empty((self.__BatchSize, self.__in_channels, *self.__dim), dtype=np.float32)
            y = np.empty((self.__BatchSize, self.__out_channels, *self.__dim), dtype=np.float32)
             
            if self.__is_val:            
                tmp_list = np.random.choice(a=np.arange(len(self.__files)), size = int(self.__BatchSize//self.__Load_per_Sample), replace=True)
            else:
                tmp_list = np.random.choice(a=np.arange(len(self.__files)), size = int(self.__BatchSize//self.__Load_per_Sample), replace=False)
            
            for i in range(self.__BatchSize):
                if i % self.__Load_per_Sample == 0:
                    self._open = self.__files[tmp_list[int(i//self.__Load_per_Sample)]]
                X[i,...], y[i,...] = self.__data_generation()
        
        elif self.__BatchSize == 1:
            self._open = self.__files[index]
            X, y = self.__data_generation()

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

            
            
    def __data_augmentation(self, image:np.ndarray, masks:np.ndarray)->dict:
        """Perform data augmentation as built in the Albumentations library.

        Args:
            image (np.ndarray): Single-color, multi-structure image
            masks (np.ndarray): Stack of ground truth structures

        Returns:
            dict: Augmented image and masks
        """
        tmp = self.__transform(image=np.squeeze(image, axis = 0), masks = [np.squeeze(a,axis = 0) for a in np.split(masks,indices_or_sections = self.__out_channels, axis = 0)])
        
        return tmp['image'], tmp['masks']

    def __data_generation(self)->np.ndarray:
        """Generates data containing # batch_size augmented samples and likewise augmented ground truth.

        Returns:
            np.ndarray: X - augmentd sample stack, y - augmented ground truth stack
        """
        
        # Initialization
        X = np.empty((self.__in_channels, *self.__dim), dtype=np.float32)
        y = np.empty((self.__out_channels, *self.__dim), dtype=np.float32)
                 
        tmp = np.load(self._open)
                
        # No Augmentation
        if self.__transform == None:
            self.__transform = A.Compose([A.RandomCrop(*self.__dim,always_apply=True),
                                          A.PadIfNeeded(min_height=self.__chunks, min_width=self.__chunks,p = 1, border_mode = 0)])


        img,masks = self.__data_augmentation(tmp[:self.__in_channels,...],tmp[self.__in_channels:,...])
        
        # Store sample
        if self.__normalize:
            X = img/725 ## normalize if all superimposed imags have the same intensity distribution
            
        elif self.__standardize:
            img = torch.from_numpy(img.astype(np.float32))
            img = ((img - img.mean())/img.std())
            X = torch.nan_to_num(img)
        
        elif not self.__normalize and not self.__standardize:
            X = img
        
        if self.__in_channels == 1:
            X = X[None,...]

        # Store class
        masks = np.stack(masks,axis=0)
        if self.__normalize or self.__standardize:
            y = masks/255
                        
        elif not self.__normalize and not self.__standardize:
            y = masks
        
        if self.__out_channels == 1:
            y = y[None,...]

        return X,y
    
def Build_Loaders(raw_source:str, 
                  val_source:str = None, 
                  val_split:float=0.1, 
                  dim=(256,256),
                  val_dim=(3000,3000),
                  in_channels=1, 
                  out_channels=3,
                  raw_aug:A.Compose = None,
                  val_aug:A.Compose = None,
                  chunks:int = 8,
                  BatchSize:int = 32,
                  shuffle:bool = True,
                  normalize:bool = False,
                  standardize:bool = False,
                  num_workers:int = 6,
                  prefetch:int = 8,
                  autobatch:bool = True,
                  val_test_batch: int = 1)->DataLoader:
    """Setup DataLoad objects and feed them with data. These objects are used to continuously generate batches for both training and validation. Multi processing with persistent workers is highly recommended.

    Args:
        raw_source (str): Path to raw data.
        val_source (str, optional): Path to validation data if applicable. Defaults to None.
        val_split (float, optional): Percentage of raw data to use as a substitution for validation. This is NOT recommended, but better than no validation when heavy augmentation is applied to the data. Defaults to 0.1.
        dim (tuple, optional): Set raw patch dimensions in px. (row,column). Defaults to (256,256).
        val_dim (tuple, optional): Set validation patch dimensions in px. (row,column). Defaults to (3000,3000).
        in_channels (int, optional): Number of input channels. Each channel is used for one feature only. (e.g. gray scale image -> 1 in-channel). Defaults to 1.
        out_channels (int, optional): Number of output channels. Is equal to the number of structures to separate. Defaults to 3.
        raw_aug (A.Compose, optional): Albumentations.Composed raw data augmentation pipeline. Defaults to None.
        val_aug (A.Compose, optional): Albumentations.Composed validation data augmentation pipeline. Defaults to None.
        chunks (int, optional): Number of patches taken from the same dataset. Defaults to 8.
        BatchSize (int, optional): Number of patches per batch. Defaults to 32.
        shuffle (bool, optional): Flag to shuffle datasets between batches. Recommended. Defaults to True.
        normalize (bool, optional): Normalize Data y/n. Defaults to False.
        standardize (bool, optional): Standardize Data (Zero-Mean, Unit-Variance) y/n. Defaults to False.
        num_workers (int, optional): Set the number of persistent workers when multiprocessing. Defaults to 6.
        prefetch (int, optional): Set how many batches should be queued per worker. Defaults to 8.
        autobatch (bool, optional): Flag if you want to use the autobatching. Might increase performance. Defaults to True.
        val_test_batch (int, optional): Set the number of batches used in validation. Defaults to 1.

    Returns:
        DataLoader: _description_
    """
    
    if val_source == None:
        if (val_split < 1.0) and ( val_split > 0.0):
            val_source = np.random.choice(raw_source, size = int(len(raw_source)*val_split), replace = False) #test_source
            raw_source = [x for x in raw_source if x not in val_source]
    
    ## Retrun DataGenerator objects for train and test data
    
    if autobatch:
        train_dataset = DataGenerator(raw_source, dim, in_channels, out_channels, raw_aug, chunks, normalize, standardize, batchsize=1)
        val_dataset = DataGenerator(val_source, val_dim, in_channels, out_channels, val_aug, chunks, normalize, standardize, batchsize=1)
                    
        return (
            DataLoader(train_dataset, batch_size=BatchSize, shuffle=shuffle, pin_memory=True, num_workers=num_workers-2, prefetch_factor = prefetch, persistent_workers = True),
            DataLoader(val_dataset, batch_size=val_test_batch, shuffle=False, num_workers=2, prefetch_factor = 4, persistent_workers = True),
            )
        
    if not autobatch:
        train_dataset = DataGenerator(raw_source, dim, in_channels, out_channels, raw_aug, chunks, normalize, standardize, batchsize=BatchSize)
        val_dataset = DataGenerator(val_source, val_dim, in_channels, out_channels, val_aug, chunks, normalize, standardize, batchsize=val_test_batch, is_val=True)
        
        if val_test_batch == 1:
                        
            return (
                DataLoader(train_dataset, batch_size=None, shuffle=shuffle, pin_memory=True, num_workers=num_workers-2, prefetch_factor = prefetch, persistent_workers = True),
                DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, prefetch_factor = 4, persistent_workers = True),
                )
        
        else:
            
            return (
                DataLoader(train_dataset, batch_size=None, shuffle=shuffle, pin_memory=True, num_workers=num_workers, prefetch_factor = prefetch),
                DataLoader(val_dataset, batch_size=None, shuffle=False, num_workers=2, prefetch_factor = 4),
                )
            
    #%% Covnenience Functions
    @staticmethod
    def generate_config():
        """Generate a configuration file for the data generator object. This file can be used to store the configuration of the data generator object for later use.

        Returns:
            dict: Configuration file
        """
        config = {
            raw_source:None,
            val_source:None, 
            val_split:0.1, 
            dim:(256,256),
            val_dim:(3000,3000),
            in_channels:1, 
            out_channels:3,
            raw_aug: None,
            val_aug: None,
            chunks: 8,
            BatchSize: 32,
            shuffle: True,
            normalize: False,
            standardize: False,
            num_workers: 6,
            prefetch: 8,
            autobatch: True,
            val_test_batch: 1
            }
        
        return config