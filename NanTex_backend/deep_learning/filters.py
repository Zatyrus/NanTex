## Dependencies
import numpy as np
import abc

# for progress bar
#detect jupyter notebook
from IPython import get_ipython
try:
    ipy_str = str(type(get_ipython()))
    if 'zmqshell' in ipy_str:
        from tqdm.notebook import tqdm, trange
    else:
        from tqdm import tqdm, trange
except:
    from tqdm import tqdm, trange


class Filter(abc.ABC):
    def __init__(self, x_size, y_size) -> None:
        self._size = (y_size, x_size)
        self._col_array = []
        self._output = 0
        
    def grab(self, value) -> None:
        self._col_array.append(value)
        
    def check_size(self) -> bool:
        if len(self._col_array) == self._size[0]*self._size[1]:
            return True
        else:
            return False
    
    def reset(self) -> None:
        self._col_array = []
    
    @abc.abstractmethod    
    def filter(self):
        pass
    
class Minimum(Filter):
    def __init__(self, x_size, y_size) -> None:
        super().__init__(x_size, y_size)
    
    def filter(self) -> None:
        self._output = np.amin(np.array(self._col_array))
        self.reset()
    
class Maximum(Filter):
    def __init__(self, x_size, y_size) -> None:
        super().__init__(x_size, y_size)
        
    def filter(self) -> None:
        self._output = np.amax(np.array(self._col_array))
        self.reset()
    
class Median(Filter):
    def __init__(self, x_size, y_size) -> None:
        super().__init__(x_size, y_size)
        
    def filter(self) -> None:
        self._output = np.median(np.array(self._col_array))
        self.reset()

class MaxThreshold(Filter):
    def __init__(self, x_size, y_size, max_th) -> None:
        super().__init__(x_size, y_size)
        self.__max_th = max_th
        
    def filter(self):
        if np.max(self._col_array) >= self.__max_th:
            self._output = self._col_array[len(self._col_array)//2]
        else:
            self._output = 0
        self.reset()
        
#%% Execution

def rank_filter(image, w:tuple, filter_mode:str, max_th:float = 0):
    filtered_image = np.zeros_like(image)
    
    filter_key = {"max":Maximum(w[0],w[1]), "med":Median(w[0], w[1]), "min": Minimum(w[0], w[1]), 'maxTH': MaxThreshold(w[0], w[1], max_th)}
    rank_filter = filter_key[filter_mode]
    
    pad_v = int((w[0]-1)/2)
    pad_h = int((w[1]-1)/2)
    img_tmp = pad_image(image=image, pad_v=pad_v, pad_h=pad_h)
    
    for i in trange(pad_v, img_tmp.shape[0]-pad_v, desc="Applying "+filter_mode+"-Filter..."):
        for j in range(pad_h, img_tmp.shape[1]-pad_h):
            for m in range(-pad_v, pad_v+1):
                for n in range(-pad_h, pad_h+1):
                    rank_filter.grab(img_tmp[i+m, j+n])
            try:   
                if rank_filter.check_size():
                    rank_filter.filter()
                    filtered_image[i-pad_v,j-pad_h] = rank_filter._output
                else:
                    raise ValueError("Value count does not match expectation.")
            except:
                return print("Please check the self._col_array for potential errors.")
    return filtered_image

        
#%% Helper Functions

# Define Image padding function
# Designed generically for even and odd paddings

def pad_image(image, pad_v:int, pad_h:int):
    img_shape = (image.shape[0]+2 * pad_v, image.shape[1]+2 * pad_h)
    img_tmp = np.zeros(shape=img_shape)
    img_tmp[pad_v:-pad_v, pad_h:-pad_h] = image
    print("IMAGE SHAPE: {} | PADDED IMAGE SHAPE: {} | PADDING SHAPE: {}".format(image.shape,img_tmp.shape, (pad_v,pad_h)))
    return img_tmp
    