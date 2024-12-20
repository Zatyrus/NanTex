## Dependencies
import os
import pathlib as pl
import numpy as np
from skimage import io
from abc import ABC, abstractmethod
from typing import Dict, List, NoReturn, Callable

## Custom Dependencies
from ..Util.pyDialogue import pyDialogue as pD

#%% File Handler
class FileHandlerCore(ABC):
    
    data_paths_in:Dict[str, List[str]]
    data_in:Dict[str, np.ndarray]
    data_path_out:str

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
        for key, path in self.data_paths_in.items():
            # path is a list of files
            if self.__is_iterable__(path) and not isinstance(path, str):
                self.data_in[key] = [self.__load_factory__(p)(p) for p in path]
                continue
            # path is a single file
            self.data_in[key] = self.__load_factory__(path)(path)
    
    def __load_factory__(self, file:str) -> Callable:
        match file.split('.')[-1]:
            case 'npy':
                return self.__load_npy__
            case 'png' | 'jpg' | 'jpeg':
                return self.__load_img__
            case _:
                return self.__load_catch__
    
    #%% Data Loading Functions
    def __load_npy__(self, file_path:str) -> np.ndarray:
        if self.DEBUG:
            print(f"Loading npy data from {file_path}...")
        return np.load(file_path)
            
    def __load_img__(self, file_path:str) -> np.ndarray:
        if self.DEBUG:
            print(f"Loading image data from {file_path}...")
        return io.imread(file_path)
    
    def __load_catch__(self, *args, **kwargs) -> NoReturn:
        if self.DEBUG:
            try:
                print(f"Skipping {args}...")
                print(f"Skipping {kwargs['file_path']}...")
            except:
                print('Skipping...')
        raise Warning('Data type not supported yet...')

    #%% Path Handling
    def __call_outpath__(self)->str:
        if (self.data_path_out == None) or (self.data_path_out == 'null') or (self.data_path_out == ''):
            return None
        pl.Path(f"{self.data_path_out}/{self.mode}").mkdir(exist_ok = True, parents = True)
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
    
    #%% Helper
    def __is_iterable__(self, 
                       obj:object
                       )->bool:
        try:
            iter(obj)
            return True
        except:
            return False
    
    #%% Metadata Handling
    @abstractmethod
    def __setup_metadata__(self)->NoReturn:
        pass
    
    #%% Abstract Methods
    @abstractmethod
    def from_explorer(self)->NoReturn:
        pass
    
    @abstractmethod
    def from_glob(self)->NoReturn:
        pass
            