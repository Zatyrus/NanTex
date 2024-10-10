## Dependencies
import os
import warnings
import numpy as np
from skimage import io
from abc import ABC, abstractmethod
from typing import Dict, List, NoReturn

## Custom Dependencies
from ..Util.pyDialogue import pyDialogue as pD

#%% File Handler
class FileHandlerCore(ABC):
    
    data_paths_in:Dict[str, List[str]]
    data_in:Dict[str, np.ndarray]
    data_path_out:str
    data_type:str

    DEBUG:bool
    mode:str
    
    
    def __post_init__(self)->NoReturn:
        try:
            self.__load_data__()
            self.__setup_metadata__()
            self.data_path_out = self.__check_outpath__()
            
        except Exception as e:
            print(f'Error: {e}')
                 
    #%% Data Loading
    def __load_data__(self) -> None:
        if self.DEBUG:
            print('Loading data...')
        ## check for datatype
        if self.data_type == 'npy':
            self.__load_npy__()
        elif self.data_type in ['png', 'jpg', 'jpeg']:
            self.__load_img__()
        else:
            warnings.warn('Data type not supported yet...')
    
    def __load_npy__(self) -> None:
        if self.DEBUG:
            print('Loading npy data...')
            
        for key, value in self.data_paths_in.items():
            self.data_in[key] = [np.load(path) for path in value]
            
    def __load_img__(self) -> None:
        if self.DEBUG:
            print('Loading image data...')
        
        for key, value in self.data_paths_in.items():
            self.data_in[key] = [io.imread(path) for path in value]

    #%% Path Handling
    def __call_outpath__(self)->str:
        if self.data_path_out == None:
            return None
        return f"{self.data_path_out}/{self.mode}"
    
    def __retrieve_outpath__(self)->str:
        if self.DEBUG:
            print('Retrieving outpath...')
        return pD.askDIR(query_title = f"Enter the output path for {self.mode} data: ")
    
    def __check_outpath__(self)->str:
        if self.DEBUG:
            print('Checking outpath...')
        outpath = self.__call_outpath__()
        if outpath == None:
            outpath = self.__retrieve_outpath__()
        return outpath
    
    def __check_filepath__(self, 
                           path:str,
                           query:str
                           )->str:
        if self.DEBUG:
            print('Checking path...')
        if os.path.isfile(self.__none_to_null__(path)):
            return path
        return pD.askFILE(query_title = f"Please select a {query} file.")
    
    def __check_filepaths__(self, 
                            paths:List[str],
                            query:str
                            )->List[str]:
        if self.DEBUG:
            print('Checking filepaths...')
        if all([os.path.isfile(path) for path in self.__none_to_null__(paths)]):
            return paths
        return pD.askFILES(query_title = f"Please select the {query} files.")
    
    def __check_dirpath__(self, 
                          path:str,
                          query:str
                          )->str:
        if self.DEBUG:
            print('Checking directory...')
        if os.path.isdir(self.__none_to_null__(path)):
            return path
        return pD.askDIR(query_title = f"Please select a {query} directory.")
    
    def __none_to_null__(self, 
                         val:str
                         )->str:
        if val == None:
            return 'null'
        return val
    
    #%% Metadata Handling
    @abstractmethod
    def __setup_metadata__(self)->NoReturn:
        pass
            