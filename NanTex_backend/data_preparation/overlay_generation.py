## Dependencies:
import sys
import ray
import time
import psutil
import warnings
import itertools
import webbrowser
import numpy as np
from typing import List, Tuple, Union, Dict, Any, Optional, NoReturn, Callable

# for progress bar
#detect jupyter notebook
from IPython import get_ipython
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


class OVERLAY_HELPER:
    #%% Overlay Generation
    @staticmethod
    def __overlay__(img_list:List[np.ndarray])->np.ndarray:
        return np.sum(img_list, axis=0)    

    @staticmethod
    def __generate_stack__(punchcard:Dict[str, Tuple[int, int]],
                           data_in:Dict[str, List[np.ndarray]]
                           )->NoReturn:
        
        # collect imgs
        out:List
        out = [data_in[key][value] for key, value in punchcard.items()]
        
        # overlay imgs
        out.append(OVERLAY_HELPER.__overlay__(out))
        return np.stack(out, axis=0)
    
    @staticmethod
    def __generate_stack_rotation__(punchcard:Dict[str, Tuple[int, int]], 
                                    data_in:Dict[str, List[np.ndarray]]
                                    )->NoReturn:
            
            # collect imgs
            out:List
            out = [data_in[key][value] for key, value in punchcard.items() if key != 'rotation']
            
            # rotate imgs
            out = [np.rot90(img, k = punchcard['rotation'][i]) for i, img in enumerate(out)]
            
            # overlay imgs
            out.append(OVERLAY_HELPER.__overlay__(out))
            return np.stack(out, axis=0)
    
    @staticmethod
    @ray.remote(num_returns=1)
    def __multi_core_worker_generate_stack__(punchcard:Dict[str, Tuple[int, int]], 
                                             data_in:Dict[str, List[np.ndarray]],
                                             data_path_out:str
                                            )->str:
        # read punchcard
        key, punchcard = list(punchcard.items())[0]
        np.save(f"{data_path_out}/{key}.npy", OVERLAY_HELPER.__generate_stack__(punchcard = punchcard,
                                                                                data_in = data_in))
        
        return key
    
    @staticmethod
    @ray.remote(num_returns=1)
    def __multi_core_worker_generate_stack_rotation__(punchcard:Dict[str, Tuple[int, int]], 
                                                      data_in:Dict[str, List[np.ndarray]],
                                                      data_path_out:str
                                                      )->str:
            # read punchcard
            key, punchcard = list(punchcard.items())[0]
            np.save(f"{data_path_out}/{key}.npy", OVERLAY_HELPER.__generate_stack_rotation__(punchcard = punchcard,
                                                                                             data_in = data_in))
            
            return key


class OverlayGenerator(FileHandlerCore):
    data_paths_in:Dict[str, List[str]]                      # key: data name, value: list of paths
    data_path_out:str                                       # key: outpath
    data_in:Dict[str, List[np.ndarray]]                     # key: data name, value: data <- img_stack
    data_punchcards:Dict[str, Dict[str, Tuple[int, int]]]   # key: data name, value: dict of data <- input_key, mod_instructions
    
    mode:str
    multi_core:bool
    data_type:str
    metadata:Dict[str, Any]
    DEBUG:bool
    
    cluster_context:Optional[Any]                           # ray cluster context       
    
    def __init__(self, 
                 data_paths_in:Dict[str, List[str]],
                 data_path_out:str = None, 
                 mode:str = 'overlay', 
                 data_type:str = 'npy',
                 multi_core:bool = False,
                 DEBUG:bool = False
                 ) -> None:
        # data variables
        self.data_paths_in = data_paths_in
        self.data_path_out = data_path_out
        self.data_in = {}
        self.data_punchcards = {}
        
        # control variables
        self.mode = mode
        self.multi_core = multi_core
        self.data_type = data_type
        self.DEBUG = DEBUG
        
        # internal variables
        self.metadata = {}
        
        # Call from parent class
        self.__post_init__()
    
    #%% classmethods
    @classmethod
    def from_explorer(cls, **kwargs) -> 'OverlayGenerator':
        data_paths_in:Dict[str, List[str]] = {}
        feature_count:int = 1
        
        ## Feature File Selection
        
        while True:
            print(f'Please select files for feature #{feature_count}...')
            inp:Union[str, tuple] = pD.askFILES(query_title = f'Please select files for feature #{feature_count}...')
            
            if inp == '':
                if feature_count == 1:
                    print('No features selected...')
                    break
                print('No more features to select...')
                break
            
            # add to dict
            data_paths_in.update({f'feature_{feature_count}':list(inp)})
            
            # increment feature count
            feature_count += 1
        
        return cls(data_paths_in = data_paths_in, **kwargs)
    
    #%% Main Functions
    def generate_punchcards(self)->NoReturn:
        ## Reset punchcards
        self.data_punchcards = {}
        
        ## generate punchcards
        if self.DEBUG:
            print('Generating punchcards...')
        
        if self.mode == 'overlay':
            self.__generate_punchcard_overlay__()
        elif self.mode == 'rotation':
            self.__generate_punchcard_overlay_rotation__()
        else:
            warnings.warn('Mode not supported yet...')
    
    def generate_overlay(self, 
                         patchsize:Tuple[int, int],
                         max_val:Union[float, int] = 255
                         )->NoReturn:        
        
        ## pad input images
        self.__pad_input_imgs__(patchsize = patchsize)
        
        ## normalize input images
        self.__normalize_input_imgs__(max_val = max_val)
        
        ## generate punchcards
        self.generate_punchcards()
        
        ## generate overlay
        if self.DEBUG:
            print('Generating overlay...')
        
        if self.multi_core:
            if self.__check_ray_status__():
                self.__run_multi_core__()
            else:
                print('Ray not initialized...')
                print("Please run 'OverlayGenerator.setup_multi_core()' to initialize Ray...") 
        else:
            self.__run_single_core__()
        
        
    def setup_multi_core(self,
                         num_cpu:int = psutil.cpu_count(logical = False),
                         num_gpu:int = 0,
                         launch_dashboard:bool = False
                         )->NoReturn:
        
        self.__setup_ray__(num_cpu = num_cpu,
                           num_gpu = num_gpu,
                           launch_dashboard = launch_dashboard)
    
    #%% Data Pre-Processing
    def __generate_stencil__(self)->NoReturn:
        tmp_ovl:list = [range(i) for i in [len(self.data_in[key]) for key in self.data_in.keys()]]
        tmp_ovl:list = list(itertools.product(*tmp_ovl))
        return tmp_ovl
    
    def __generate_stencil_rotation__(self)->NoReturn:
        tmp_ovl:list = [range(i) for i in [len(self.data_in[key]) for key in self.data_in.keys()]]
        rot_ovl:list = [range(4) for _ in range(len(self.data_in.keys()))]
        tmp_ovl += rot_ovl
        
        tmp_ovl:list = list(itertools.product(*tmp_ovl))
        return tmp_ovl
    
    def __generate_punchcard_overlay__(self)->NoReturn:
        for punchcard in self.__generate_stencil__():
            self.data_punchcards[f"{len(self.data_in.keys())}_feature_{self.mode}_{punchcard}"] = {list(self.data_in.keys())[i]:punchcard[i] for i in range(len(self.data_in.keys()))}

    def __generate_punchcard_overlay_rotation__(self)->NoReturn:
        for punchcard in self.__generate_stencil_rotation__():
            self.data_punchcards[f"{len(self.data_in.keys())}_feature_{self.mode}_{punchcard}"] = {list(self.data_in.keys())[i]:punchcard[i] for i in range(len(self.data_in.keys()))}
            self.data_punchcards[f"{len(self.data_in.keys())}_feature_{self.mode}_{punchcard}"].update({'rotation':punchcard[len(self.data_in.keys()):]})

    #%% Single Core Execution
    def __single_core_main__(self, stack_generation:Callable, desc:str)->NoReturn:
        with tqdm(total=len(self.data_punchcards.keys()), desc = desc, file=sys.stdout) as pbar:
            # iterate over punchcards
            for key, punchcard in self.data_punchcards.items():
                np.save(f"{self.data_path_out}/{key}.npy", stack_generation(punchcard = punchcard,
                                                                            data_in = self.data_in))
                
                ## update progress bar
                pbar.update(1)
            
            # handle pbar
            pbar.colour = 'green'
            pbar.close()
    
    def __run_single_core__(self)->NoReturn:
        if self.mode == 'overlay':
            self.__single_core_main__(stack_generation = OVERLAY_HELPER.__generate_stack__, 
                                      desc='Generating Single Core Overlays...')
        elif self.mode == 'rotation':
            self.__single_core_main__(stack_generation = OVERLAY_HELPER.__generate_stack_rotation__, 
                                      desc='Generating Single Core Rotational Overlays...')
        else:
            warnings.warn('Mode not supported yet...')
    
    #%% Multi Core Execution
    def __setup_ray__(self,
                      num_cpu:int,
                      num_gpu:int,
                      launch_dashboard:bool)->NoReturn:
    
        if self.DEBUG:
            print('Setting up Ray...')
        
        # shutdown any stray ray instances
        ray.shutdown()
        
        # ray init
        cluster_context = ray.init(num_cpus = num_cpu, 
                                   num_gpus = num_gpu,
                                   ignore_reinit_error=True)
        
        # dashboard
        if launch_dashboard:
            try:
                webbrowser.get('windows-default').open(f"http://{cluster_context.dashboard_url}",
                                                       autoraise = True,
                                                       new = 2)
            except Exception as e:
                print(f'Error: {e}')
        
        if self.DEBUG:
            print('Ray setup complete...')
            print(f'Ray Dashboard: {cluster_context.dashboard_url}')
        self._ray_instance = cluster_context
            
    def __shutdown_ray__(self)->NoReturn:
        if self.DEBUG:
            print('Shutting down Ray...')
        ray.shutdown()
        
    def __offload_punchcards_to_ray__(self)->List[ray.ObjectRef]:
        if self.DEBUG:
            print('Offloading punchcards to Ray...')
        return [ray.put({key:punchcard}) for key, punchcard in self.data_punchcards.items()]
    
    def __offload_data_in_to_ray__(self)->List[ray.ObjectRef]:
        if self.DEBUG:
            print('Offloading data_in to Ray...')
        return ray.put(self.data_in)
    
    def __offload_data_path_out_to_ray__(self)->ray.ObjectRef:
        if self.DEBUG:
            print('Offloading data_outpath to Ray...')
        return ray.put(self.data_path_out)
                
    def __multi_core_main__(self,
                            worker:Callable,
                            sleep_time:float)->NoReturn:        
        ## offload data
        data_in_ref = self.__offload_data_in_to_ray__()
        punchcard_refs = self.__offload_punchcards_to_ray__()
        data_path_out_ref = self.__offload_data_path_out_to_ray__()
        
        ## listen to progress
        status, finished_states = self.__listen_to_ray_progress__(
            object_references=[worker.remote(punchcard = punchcard,
                                             data_in = data_in_ref,
                                             data_path_out = data_path_out_ref)
                               for punchcard in tqdm(punchcard_refs, 
                                                     desc="Scheduling Workers", 
                                                     position = 0)],
            total = len(punchcard_refs),
            sleep_time = sleep_time)
        
        ## check completion
        if status:
            try:
                assert len(finished_states) == len(punchcard_refs)
                assert all([key in finished_states for key in self.data_punchcards.keys()])
            except Exception as e:
                print(f'Error: {e}')
                
        ## Shutdown Ray
        print("Multi Core Execution Complete...")
        print("Use 'OverlayGenerator.shutdown_multi_core()' to shutdown the cluster.")
        
    def __run_multi_core__(self)->NoReturn:
        ## run multi core
        if self.mode == 'overlay':
            self.__multi_core_main__(worker = OVERLAY_HELPER.__multi_core_worker_generate_stack__,
                                     sleep_time = self.metadata['sleeptime'])
        elif self.mode == 'rotation':
            self.__multi_core_main__(worker = OVERLAY_HELPER.__multi_core_worker_generate_stack_rotation__,
                                     sleep_time = self.metadata['sleeptime'])
        else:
            warnings.warn('Mode not supported yet...')
            
        
    #%% Helper Functions
    def __pad_img__(self, 
                img:np.ndarray,
                y_lim:int,
                x_lim:int,
                patchsize:Tuple[int, int] = None
                )->np.ndarray:
        # determine padding
        y_pad = max(0,int((y_lim-img.shape[0])/2))
        x_pad = max(0,int((x_lim-img.shape[1])/2))
        
        # setup zero array
        out = np.zeros(shape=(y_lim,x_lim))
        # insert image
        out[y_pad:img.shape[0] + y_pad, x_pad:img.shape[1] + x_pad] = img
        
        # crop to limit
        if patchsize:
            out = out[int((y_lim-patchsize[0])//2):-(int((y_lim-patchsize[0])//2)+1),int((x_lim-patchsize[1])//2):-(int((x_lim-patchsize[1])//2)+1)]
            return out
        # return padded image
        return out
    
    def __normalize_img__(self, 
                      img:np.ndarray,
                      max_val:Union[float, int] = 255
                      )->np.ndarray:
        return (img - np.min(img))/(np.max(img)-np.min(img)) * max_val

    def get_x_lim(self)->int:
        return max([img.shape[1] for img in sum(list(self.data_in.values()),[])])
    
    def get_y_lim(self)->int:
        return max([img.shape[0] for img in sum(list(self.data_in.values()),[])])

    def __listen_to_ray_progress__(self,
                                   object_references:List[ray.ObjectRef], 
                                   total:int, 
                                   sleep_time:float = 0.1)->bool:
        if self.DEBUG:
            print('Setting up progress monitors...')
        ## create progress monitors
        msd_progress = tqdm(total = total, desc = 'Workers', position = 1)
        cpu_progress = tqdm(total = 100, desc="CPU usage", bar_format='{desc}: {percentage:3.0f}%|{bar}|', position = 2)
        mem_progress = tqdm(total=psutil.virtual_memory().total, desc="RAM usage", bar_format='{desc}: {percentage:3.0f}%|{bar}|', position = 3)
        
        finished_states = []
        
        if self.DEBUG:
            print('Listening to Ray Progress...')
        ## listen for progress
        while len(object_references) > 0:
            try:
                # get the ready refs
                finished, object_references = ray.wait(
                    object_references, timeout=8.0
                )
                
                data = ray.get(finished)
                finished_states.extend(data)
                
                # update the progress bars
                mem_progress.n = psutil.virtual_memory().used
                mem_progress.refresh()
                
                cpu_progress.n = psutil.cpu_percent()
                cpu_progress.refresh()
                
                # update the progress bar
                msd_progress.n = len(finished_states)
                msd_progress.refresh()
            
                # sleep for a bit
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                print('Interrupted')
                break
        
        # set the progress bars to success
        msd_progress.colour = 'green'
        cpu_progress.colour = 'green'
        mem_progress.colour = 'green'
        
        # set the progress bars to their final values
        msd_progress.n = total
        cpu_progress.n = 0
        mem_progress.n = 0
        
        # close the progress bars
        msd_progress.close()
        cpu_progress.close()
        mem_progress.close()
        
        if self.DEBUG:
            print('Ray Progress Complete...')
        
        return True, finished_states
    
    def __check_ray_status__(self)->NoReturn:
        if self.DEBUG:
            print('Checking Ray Status...')
        return ray.is_initialized()
    
    def shutdown_multi_core(self)->NoReturn:
        self.__shutdown_ray__()
    
    def __reset__(self)->NoReturn:
        self.data_in = {}
        self.data_punchcards = {}
        self.metadata = {}
        self.data_path_out = None
        
        try:
            self.__shutdown_ray__()
        except Exception as e:
            print(f'Error: {e}')
        
    def _reboot_(self)->NoReturn:
        self.__reset__()
        self.__post_init__()

    #%% Pre-Processing
    def __pad_input_imgs__(self, patchsize:Tuple[int, int])->NoReturn:
        if self.DEBUG:
            print('Padding input images...')
        for key, value in self.data_in.items():
            self.data_in[key] = [self.__pad_img__(img = img, 
                                              y_lim = self.metadata['y_lim'], 
                                              x_lim = self.metadata['x_lim'], 
                                              patchsize = patchsize) for img in value]

    def __normalize_input_imgs__(self, max_val:Union[float, int])->NoReturn:
        if self.DEBUG:
            print('Normalizing input images...')
        for key, value in self.data_in.items():
            self.data_in[key] = [self.__normalize_img__(img = img, max_val = max_val) for img in value]
    
    #%% Metadata Handling
    def __setup_metadata__(self)->NoReturn:
        if self.DEBUG:
            print('Setting up metadata...')
        self.metadata = {
            'x_lim':self.get_x_lim(),
            'y_lim':self.get_y_lim(),
            'sleeptime':0.0
        }