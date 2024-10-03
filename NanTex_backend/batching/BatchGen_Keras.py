########################################################################
##### BATCH GENERATOR written in Keras API <- currently not in use #####
########################################################################

## Dpendencies
import numpy as np
import keras

class BatchGenerator(keras.utils.Sequence):
    'Load, Augment and distribute batches for training & valitation.'
    def __init__(self, files, batch_size=32, dim=(256,256), in_channels=1, out_channels=3,
                 shuffle=True, aug_line:A.Compose = None, num_per_sample:int = 8, StepOnEpoch:int = 100):
        'Initialization'
        self.__dim = dim
        self.batch_size = batch_size
        self.__num_per_sample = num_per_sample
        self.__files = files
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__shuffle = shuffle
        self.__transform = aug_line
        self.__StepOnEpoch = StepOnEpoch
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.__StepOnEpoch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate data
        X, y = self.__data_generation()

        return X, y

    def on_epoch_end(self):
        'Updates indices after each epoch'
        if self.__shuffle == True:
            self._open = np.random.choice(self.__files, size= int(self.batch_size//self.__num_per_sample))
        else:
            self._open = self.__files[:int(self.batch_size//self.__num_per_sample)]
            
    def __data_augmentation(self, image, masks):
                
        tmp = self.__transform(image=image, masks = np.split(masks,indices_or_sections = self.__out_channels, axis = 2))
        
        return tmp['image'], tmp['masks']

    def __data_generation(self):
        'Generates data containing # batch_size samples'
        
        # Initialization
        X = np.empty((self.batch_size, *self.__dim, self.__in_channels), dtype=np.float32)
        y = np.empty((self.batch_size, *self.__dim, self.__out_channels), dtype=np.float32)
        
        # Augmentation
        if self.__transform == None:
            for i, file in enumerate(self._open):
                
                for j in range(self.__num_per_sample):
                    # Store sample
                    X[(self.__num_per_sample*i)+j,...] = np.load(file)[:self.__dim[0], :self.__dim[1],:self.__in_channels]

                    # Store class
                    y[(self.__num_per_sample*i)+j,...] = np.load(file)[:self.__dim[0], :self.__dim[1],self.__in_channels:]
            
            if self.__shuffle:
                return sk.utils.shuffle(X,y)
            else:
                return X,y

        if self.__transform != None:
            for i, file in enumerate(self._open):
                
                # Augmentation
                for j in range(self.__num_per_sample):
                    tmp = np.load(file)
                    img,masks = self.__data_augmentation(tmp[...,:self.__in_channels],tmp[...,self.__in_channels:])
                    
                    # Store sample
                    X[(self.__num_per_sample*i)+j,...] = img/np.max(img)

                    # Store class
                    masks = np.stack(masks,axis=2)[...,0]
                    y[(self.__num_per_sample*i)+j,...] = masks/np.max(masks)
            
            if self.__shuffle:
                return sk.utils.shuffle(X,y)
            else:
                return X,y
        
        else:
            raise(TypeError('No valid strategy was chosen!!!')) 