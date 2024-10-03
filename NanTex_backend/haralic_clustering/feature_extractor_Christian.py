# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 14:29:47 2022

@author: Gregor_Gentsch
"""

import numpy as np
from PIL import Image
import mahotas as mh
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import os
import tifffile
import ray
plt.style.use('dark_background')



@ray.remote
def feature_extractor(kernelsize, image_location, name_savefile):#, save_directory,  padding=False, output='.npy'):
    """
    ----------
    kernelsize : int (Integer)
        Size of the kernel, that walks over the image - usually uneven (e.g. 3 5 7 9 and so on).
    image_location : str (String)
        Image location, for example D:/WUE2022/Mictub.tif .
    name_savefile : str (String)
        Name you want to give the savefile, for example Mictub7x7 (no data type ending!)
    save_directory : str (String)
        Save generated feature data in this directory, for example D:/Analysis .
    padding : Boolean (True or false), optional
        Adds a boundary of zeroes around the feature maps to match the dimensions of the original image. The default is False.
    output : str (String, either '.npy' or '.tif' at the moment), optional
        Returns generated feature maps either as .npy data to load with numpy.load(), or as tif-stack to import in ImageJ. The default is '.npy'.

    Returns
    -------
    None.
    """
    crop = int((kernelsize-1)/2)
    #image to be analyzed location
    img_loc = image_location
    #image to array
    img = np.load(img_loc)
    ground_truth = img[1:,...]
    img = img[0,...]
    #cut image down to size
    cropimg = img[crop:-crop,crop:-crop]
    #analyze only windows with nonzero center pixel, e.g. find nonzero pixels
    nonz_index = np.array(np.nonzero(cropimg))
    #sliding window view
    stride = np.lib.stride_tricks.sliding_window_view(img,(kernelsize,kernelsize))
    #list for features to be dumped
    nonz_pix_feat = []
    #iterate over all nonzero pixels
    for i in range(0,len(nonz_index[0]),1):
        x = nonz_index[0,i]
        y = nonz_index[1,i]
        #feature computation
        feats = mh.features.haralick(stride[x,y,:,:],return_mean_ptp=True)
        nonz_pix_feat.append(feats)
    #list to array
    nonz_pix_feat = np.array(nonz_pix_feat)
    #scale features from 0 to 1, 0 is background, thats why feat range min > 0
    scaler = MinMaxScaler(feature_range=(0.0001,1))
    for i in range(0,26,1):
        scaler.fit(nonz_pix_feat[:,i].reshape(-1,1))
        nonz_pix_feat[:,i] = scaler.transform(nonz_pix_feat[:,i].reshape(-1,1)).reshape(1,-1)
    #implement padding or not
    #make target for image stack
    target = np.empty((26,len(cropimg),len(cropimg[0])))
    #fill pixels into image stack
    for i in range(0,len(nonz_index[0]),1):
            x = nonz_index[0,i] 
            y = nonz_index[1,i] 
            target[:,x,y] = nonz_pix_feat[i,:]
    
    np.save(name_savefile + ".npy", np.concatenate([img[None,crop:-crop, crop:-crop] ,target, ground_truth[:,crop:-crop, crop:-crop]], axis = 0, dtype=np.float32))
    
#     if padding == True:
#         target = np.pad(target, crop, constant_values=0)
#     elif padding == False:
#         pass
#     else :
#         print("error: Padding has to be True or False")
        
#     #concern output
#     #os.chdir(save_directory)
#     if output =='.npy':
#         np.save(name_savefile + ".npy", np.concatenate([img[None,crop:-crop, crop:-crop] ,target, ground_truth[:,crop:-crop, crop:-crop]], axis = 0, dtype=np.float32))
        
#     elif output ==".tif":
#         tifffile.imsave(name_savefile + "_tiffstack" + ".tif", target)
        
#     else:
#         print("output format not supported")
    

# #%%
# #-------------------------------------------------------------------------
# # ab hier Parallel Processing
# #modul was gebraucht wird
# from multiprocessing import Process

# #die genaue funktion dieser structur hat sich mir noch nicht ganz erschlossen, es scheint aber für die Ausführung wichtig zu sein

# if __name__ == '__main__':

#     # construct a different process for each function
#     #target ist die Funktion die ausgeführt werden soll
#     #bei args die Parameter der Funktion, ich hab mal die letzten die ich benutzt habe drin gelassen als Beispiel
#     #8 Processes weil der Rechner 8 Kerne hatte, kann man bestimmt nach oben skalieren
#     processes = [Process(target=feature_extractor, args=(7, r"F:\Bela Data\Wue_MT_647_only_1_crop.tif" ,"Wue_MT_647_only_1_crop_7x7" ,r"F:\Bela Data")),
#                  Process(target=feature_extractor, args=(7, r"F:\Bela Data\Wue_MT_647_only_5_crop.tif" ,"Wue_MT_647_only_5_crop_7x7" ,r"F:\Bela Data")),
#                  Process(target=feature_extractor, args=(7, r"F:\Bela Data\Wue_MT_647_only_9_crop.tif" ,"Wue_MT_647_only_9_crop_7x7" ,r"F:\Bela Data")),
#                  Process(target=feature_extractor, args=(7, r"F:\Bela Data\Wue_MT_clathrin_647_clathrin_only_3_crop.tif" ,"Wue_MT_clathrin_647_clathrin_only_3_crop_7x7" ,r"F:\Bela Data")),
#                  Process(target=feature_extractor, args=(7, r"F:\Bela Data\Wue_MT_clathrin_647_clathrin_only_4_crop.tif" ,"Wue_MT_clathrin_647_clathrin_only_4_crop_7x7" ,r"F:\Bela Data")),
#                  Process(target=feature_extractor, args=(7, r"F:\Bela Data\Wue_MT_clathrin_647_clathrin_only_5_crop.tif" ,"Wue_MT_clathrin_647_clathrin_only_5_crop_7x7" ,r"F:\Bela Data")),
#                  Process(target=feature_extractor, args=(7, r"F:\Bela Data\Wue_MT_clathrin_647_mixed_3_crop.tif" ,"Wue_MT_clathrin_647_mixed_3_crop_7x7" ,r"F:\Bela Data")),
#                  Process(target=feature_extractor, args=(7, r"F:\Bela Data\Wue_MT_clathrin_647_mixed_4_2_crop.tif" ,"Wue_MT_clathrin_647_mixed_4_2_crop_7x7" ,r"F:\Bela Data")),

#                          ]
#     # kick them off 
#     for process in processes:
#         process.start()
#     # now wait for them to finish
#     for process in processes:
#         process.join()
    
    



@ray.remote
def WU_feature_extractor(kernelsize, image_location, name_savefile):
    """
    ----------
    kernelsize : int (Integer)
        Size of the kernel, that walks over the image - usually uneven (e.g. 3 5 7 9 and so on).
    image_location : str (String)
        Image location, for example D:/WUE2022/Mictub.tif .
    name_savefile : str (String)
        Name you want to give the savefile, for example Mictub7x7 (no data type ending!)
    save_directory : str (String)
        Save generated feature data in this directory, for example D:/Analysis .
    padding : Boolean (True or false), optional
        Adds a boundary of zeroes around the feature maps to match the dimensions of the original image. The default is False.
    output : str (String, either '.npy' or '.tif' at the moment), optional
        Returns generated feature maps either as .npy data to load with numpy.load(), or as tif-stack to import in ImageJ. The default is '.npy'.

    Returns
    -------
    None.
    """
    crop = int((kernelsize-1)/2)
    #image to be analyzed location
    img_loc = image_location
    #image to array
    img = np.load(img_loc).astype(np.uint8)
    cut_x, cut_y = (img.shape[1]-1268)//2, (img.shape[0]-1268)//2
    img = img[cut_y:-cut_y, cut_x:-cut_x]
    #cut image down to size
    cropimg = img[crop:-crop,crop:-crop]
    #analyze only windows with nonzero center pixel, e.g. find nonzero pixels
    nonz_index = np.array(np.nonzero(cropimg))
    #sliding window view
    stride = np.lib.stride_tricks.sliding_window_view(img,(kernelsize,kernelsize))
    #list for features to be dumped
    nonz_pix_feat = []
    #iterate over all nonzero pixels
    for i in range(0,len(nonz_index[0]),1):
        x = nonz_index[0,i]
        y = nonz_index[1,i]
        #feature computation
        feats = mh.features.haralick(stride[x,y,:,:],return_mean_ptp=True)
        nonz_pix_feat.append(feats)
    #list to array
    nonz_pix_feat = np.array(nonz_pix_feat)
    #scale features from 0 to 1, 0 is background, thats why feat range min > 0
    scaler = MinMaxScaler(feature_range=(0.0001,1))
    for i in range(0,26,1):
        scaler.fit(nonz_pix_feat[:,i].reshape(-1,1))
        nonz_pix_feat[:,i] = scaler.transform(nonz_pix_feat[:,i].reshape(-1,1)).reshape(1,-1)
    #implement padding or not
    #make target for image stack
    target = np.empty((26,len(cropimg),len(cropimg[0])))
    #fill pixels into image stack
    for i in range(0,len(nonz_index[0]),1):
            x = nonz_index[0,i] 
            y = nonz_index[1,i] 
            target[:,x,y] = nonz_pix_feat[i,:]
    
    np.save(name_savefile + ".npy", np.concatenate([img[None,crop:-crop, crop:-crop] ,target], axis = 0, dtype=np.float32))
















