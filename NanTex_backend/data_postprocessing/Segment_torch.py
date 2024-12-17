from genericpath import isfile
from copy import deepcopy
import os
from tkinter.tix import NoteBook
import torch
import numpy as np
from patchify import patchify, unpatchify
import tkinter as tk
from tkinter import Image, filedialog
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage.io import imread
from NanTex_backend.evaluation.Evaluate import eval_image

def askDIR():
    root = tk.Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    DIR_path = filedialog.askdirectory()
    return DIR_path

def askFILE():
    root = tk.Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    FILE_path = filedialog.askopenfilename()
    return FILE_path

def askFILES():
    root = tk.Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    FILE_path = filedialog.askopenfilenames()
    return FILE_path

def binarize(arr,border,fill):
    arr[arr > border] = fill
    return arr

def normalize(arr):
    return arr/np.max(arr)
    
def count(arr,target):
    un, cnt = np.unique(arr, return_counts=True)
    return int(cnt[np.where(un == target)])

import torch
import torch.nn.functional as F
from math import exp
import numpy as np


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    # weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output

# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible,
        # return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average, normalize=True)


def Segment_SMLM(IMG_path:str = None, 
                 patchsize:tuple = (256,256), 
                 predictor = None, 
                 MIC_threshold:float = 0.2, 
                 LNP_threshold:float = 0.0, 
                 ACT_threshold:float = 0.2, 
                 lower_dyn_th:int = -1,
                 upper_dyn_th:int = 2, 
                 device = torch.device("cuda"),
                 activation = torch.nn.Identity(),
                 plot_results:bool = True, 
                 bin_diff:bool = True,
                 evaluate:bool = False,
                 standardize:bool = False, 
                 plot_two_channel_images:bool = False,
                 save_predictions:bool = False,
                 out_path:str = None,
                 filename:str = None):
    
    while not os.path.isfile(IMG_path):
        print('Please pick an image to segment!')
        IMG_path = askFILE()
            
    try:
        if IMG_path.endswith('.npy'):
            IMG = np.load(IMG_path) 
        elif IMG_path.endswith(('.png','.jpg','.jpeg','.tif','.tiff')):
            IMG = np.array(imread(IMG_path)).astype(np.uint16)
    except:
        return print('Expected numpy .npy or image file format. Please check your image selection!')
            
    if (IMG.shape[1] % patchsize[0] == 0) & (IMG.shape[2] % patchsize[1] == 0):
        Image_CUT = IMG.copy()[...]
        
    elif (IMG.shape[1] % patchsize[0] == 0):
        Image_CUT = IMG.copy()
        Image_CUT = Image_CUT[:,:, :-(Image_CUT.shape[2] % patchsize[1])]
        
    elif (IMG.shape[2] % patchsize[1] == 0):
        Image_CUT = IMG.copy()
        Image_CUT = Image_CUT[:,:-(Image_CUT.shape[1] % patchsize[0]),:]
        
    else:
        Image_CUT = IMG.copy()
        Image_CUT = Image_CUT[:,:-(Image_CUT.shape[1] % patchsize[0]), :-(Image_CUT.shape[2] % patchsize[1])]
        
    
    NumImg_Y = int((np.floor(Image_CUT.shape[1]/patchsize[0])))
    NumImg_X = int((np.floor(Image_CUT.shape[2]/patchsize[1])))
    NumImg_tmp = NumImg_Y*NumImg_X
    
    print("Image has shape {} and is going to be cut into {} pieces of shape {}.".format(IMG[0,:,:].shape, int(NumImg_tmp),(patchsize)))
    
    Patches = patchify(Image_CUT[0,...], patchsize, patchsize[0])
    Data_In = np.reshape(Patches,(NumImg_tmp,1,patchsize[0],patchsize[1]))
    Data_In = Data_In.astype('float32')
    
    if not standardize:
        Data_In = Data_In/725#722#728#580#725#np.max(Data_In) #255

    predictor.eval()   
    Pred = np.zeros(shape=(NumImg_tmp,3, patchsize[0], patchsize[1]))
    
    for i,patch in enumerate(tqdm(Data_In, desc='Predicting...')):
        if not np.any(patch):
            Pred[i,...] = patch
            print('SKIP')
            continue
        tmp = torch.from_numpy(patch[None,...].astype(np.float32))
        if standardize:
            tmp = ((tmp - tmp.mean())/tmp.std())
            tmp = torch.nan_to_num(tmp)
        tmp = tmp.to(device)    
        Pred[i,...] = activation(predictor(tmp)).cpu().detach().numpy()
        
    
    ####################################################################
    
    MIC = Pred[:,0,:,:]
    MIC = MIC - np.min(MIC.flatten())

    LNP = Pred[:,1,:,:]
    LNP = LNP - np.min(LNP.flatten())

    ACT = Pred[:,2,:,:]
    MIC = MIC - np.min(MIC.flatten())

    dyn_th_pre_param = lower_dyn_th
    dyn_th_post_param = upper_dyn_th
    
    from filters import rank_filter
    
    tmp_mic, bord_mic = np.histogram(MIC, bins = 100)
    MIC_th = (bord_mic[np.argmax(tmp_mic)+dyn_th_pre_param], bord_mic[np.argmax(tmp_mic)+dyn_th_post_param])
    MIC[(MIC>=MIC_th[0]) & (MIC<=MIC_th[1])] = 0
    MIC[MIC < MIC_threshold] = 0
        
    #print(MIC.shape)
    #MIC = np.array([rank_filter(sub_MIC, (5,5), 'maxTH', threshold) for sub_MIC in MIC]).reshape((MIC.shape))
    MIC_pred = unpatchify(np.reshape(MIC, (NumImg_Y,NumImg_X,patchsize[0],patchsize[1])), Image_CUT[1,:,:].shape)
    # MIC_pred = (MIC_pred - np.min(MIC_pred))
    # MIC_pred = (MIC_pred/np.max(MIC_pred))*255
    #MIC_pred = rank_filter(MIC_pred, (5,5), 'maxTH', threshold)
    #MIC = MIC/np.max(MIC.flatten())
    
    tmp_lnp, bord_lnp = np.histogram(LNP, bins = 100)
    LNP_th = (bord_lnp[np.argmax(tmp_lnp)+dyn_th_pre_param], bord_lnp[np.argmax(tmp_lnp)+dyn_th_post_param])
    LNP[(LNP>=LNP_th[0]) & (LNP<=LNP_th[1])] = 0
    LNP[LNP < LNP_threshold] = 0
    LNP_pred = unpatchify(np.reshape(LNP, (NumImg_Y,NumImg_X,patchsize[0],patchsize[1])), Image_CUT[2,:,:].shape)
    #LNP_pred = rank_filter(LNP_pred, (5,5), 'maxTH', threshold)
    #LNP = LNP/np.max(LNP.flatten())
    
    tmp_act, bord_act = np.histogram(ACT, bins = 100)
    ACT_th = (bord_act[np.argmax(tmp_act)+dyn_th_pre_param], bord_act[np.argmax(tmp_act)+dyn_th_post_param])
    ACT[(ACT>=ACT_th[0]) & (ACT<=ACT_th[1])] = 0
    ACT[ACT < ACT_threshold] = 0
    ACT_pred = unpatchify(np.reshape(ACT, (NumImg_Y,NumImg_X,patchsize[0],patchsize[1])), Image_CUT[3,:,:].shape)
    #ACT_pred = rank_filter(ACT_pred, (5,5), 'maxTH', threshold)
    #ACT = ACT/np.max(ACT.flatten())
    
    # SSIM_Metr = SSIM()
    # SSIM_Metr.to(device)
    # MSSSIM_Metr = MSSSIM()
    # MSSSIM_Metr.to(device)
    
    ###################################################################
    
    if plot_results:
        plt.style.use('dark_background')
        plt.subplots_adjust(wspace=0.3, top=1)
        font = {'family' : 'Courier new',
                'weight' : 'bold',
                'size'   : 6}

        mpl.rc('font', **font)
        
        fig, ((ax1,ax2,ax3,ax7),(ax4,ax5,ax6,ax8),(ax9,ax10,ax11,ax12),(ax13, ax14, ax15, ax16)) = plt.subplots(4,4,figsize = (15,15), dpi = 200)
            
        ax1.axis('off')
        ax1.set_title('Vanilla Image')
        im1 = ax1.imshow(Image_CUT[0,:,:], cmap = 'gray')
        plt.colorbar(im1, ax = ax1)
        
        ax2.axis('off')
        ax2.set_title('Ground truth ER')
        im2 = ax2.imshow(Image_CUT[1,...], cmap = 'plasma')
        plt.colorbar(im2, ax = ax2)

        ax3.axis('off')
        ax3.set_title('Ground truth MIC')
        im3 = ax3.imshow(Image_CUT[2,...], cmap = 'plasma')
        plt.colorbar(im3, ax = ax3)
        
        ax7.axis('off')
        ax7.set_title('Ground truth CLA')
        im7 = ax7.imshow(Image_CUT[3,...], cmap = 'plasma')
        plt.colorbar(im7, ax = ax7)

        ax4.axis('off')
        ax4.set_title('Segmented Overlay')
        im4 = ax4.imshow(MIC_pred+LNP_pred+ACT_pred, cmap = 'gray')
        plt.colorbar(im4, ax = ax4)
        
        ax5.axis('off')
        ax5.set_title('Segmented ER')
        im5 = ax5.imshow(MIC_pred, cmap = 'plasma')
        plt.colorbar(im5, ax = ax5)
        
        ax6.axis('off')
        ax6.set_title('Segmented MIC')
        im6 = ax6.imshow(LNP_pred, cmap = 'plasma')
        plt.colorbar(im6, ax = ax6)
        
        ax8.axis('off')
        ax8.set_title('Segmented CLA')
        im8 = ax8.imshow(ACT_pred, cmap = 'plasma')
        plt.colorbar(im8, ax = ax8)
        
        ax9.axis('off')
        if bin_diff:
            ovl_diff = np.abs(binarize(MIC_pred+LNP_pred+ACT_pred,0,1)-binarize(Image_CUT[0,:,:],0,1))
            ax9.set_title('Ovl Diff | diff_px: {:.2%}'.format(count(ovl_diff,1)/Image_CUT[0,:,:].size))
        else:
            ovl_diff = np.abs(normalize(MIC_pred+LNP_pred+ACT_pred)-normalize(Image_CUT[0,:,:]))
            ax9.set_title('Ovl Diff | diff_avr: {:.2e}'.format(np.average(ovl_diff)))
        im9 = ax9.imshow(ovl_diff, cmap = 'gray')
        plt.colorbar(im9, ax = ax9)
        
        ax10.axis('off')
        if bin_diff:
            MIC_diff = np.abs(binarize(MIC_pred,0,1)-binarize(Image_CUT[1,:,:],0,1))
            ax10.set_title('MIC Diff | diff_px: {:.2%}'.format(count(MIC_diff,1)/count(binarize(Image_CUT[1,:,:],0,1),1)))
        else:
            MIC_diff = np.abs(normalize(MIC_pred)-normalize(Image_CUT[1,:,:]))
            ax10.set_title('MIC Diff | diff_avr: {:.2e}'.format(np.average(MIC_diff)))
        im10 = ax10.imshow(MIC_diff, cmap = 'plasma')
        plt.colorbar(im10, ax = ax10)
        
        ax11.axis('off')
        if bin_diff:
            LNP_diff = np.abs(binarize(LNP_pred,0,1)-binarize(Image_CUT[2,:,:],0,1))
            ax11.set_title('LNP Diff | diff_px: {:.2%}'.format(count(LNP_diff,1)/count(binarize(Image_CUT[2,:,:],0,1),1)))
        else:
            LNP_diff = np.abs(normalize(LNP_pred)-normalize(Image_CUT[2,:,:]))
            ax11.set_title('LNP Diff | diff_avr: {:.2e}'.format(np.average(LNP_diff)))
        im11 = ax11.imshow(LNP_diff, cmap = 'plasma')
        plt.colorbar(im11, ax = ax11)
        
        ax12.axis('off')
        if bin_diff:
            ACT_diff = np.abs(binarize(ACT_pred,0,1)-binarize(Image_CUT[3,:,:],0,1))
            ax12.set_title('Actin Diff | diff_px: {:.2%}'.format(count(ACT_diff,1)/count(binarize(Image_CUT[3,:,:],0,1),1)))
        else:
            ACT_diff = np.abs(normalize(ACT_pred)-normalize(Image_CUT[3,:,:]))
            ax12.set_title('ACT Diff | diff_avr: {:.2e}'.format(np.average(ACT_diff)))
        im12 = ax12.imshow(ACT_diff, cmap = 'plasma')
        plt.colorbar(im12, ax = ax12)
        
        ax13.set_title('SegOL Int. HIST')
        ax13.hist(MIC.flatten()+LNP.flatten()+ACT.flatten(), bins = 100, color='orange')
        
        ax14.set_title('Seg MIC Int. HIST')
        ax14.hist(MIC.flatten(), bins = 100, color='orange')
        
        ax15.set_title('Seg LNP Int. HIST')
        ax15.hist(LNP.flatten(), bins = 100, color='orange')
        
        ax16.set_title('Seg ACT Int. HIST')
        ax16.hist(ACT.flatten(), bins = 100, color='orange')
    
    if plot_two_channel_images:
        plt.style.use('default')
        plt.figure(figsize = (15,5), dpi = 200, frameon=False)
        plt.subplot(1,3,1)
        plt.title('MIC ~ Pred(Blue) ~ Original(Orange)')
        plt.imshow(Image_CUT[1,:,:],'Oranges',interpolation='bilinear', alpha = 1)
        plt.imshow(MIC_pred,'Blues',interpolation='bilinear', alpha = 0.3)

        
        plt.subplot(1,3,2)
        plt.title('LNP ~ Pred(Blue) ~ Original(Orange)')
        plt.imshow(Image_CUT[2,:,:],'Oranges',interpolation='bilinear', alpha = 1)
        plt.imshow(LNP_pred,'Blues',interpolation='bilinear', alpha = 0.3)
        
        plt.subplot(1,3,3)
        plt.title('ACT ~ Pred(Blue) ~ Original(Orange)')
        plt.imshow(Image_CUT[3,:,:],'Oranges',interpolation='bilinear', alpha = 1)
        plt.imshow(ACT_pred,'Blues',interpolation='bilinear', alpha = 0.3)
    
    if evaluate:
        print('Evaluating...')
        print(Image_CUT[1,...].shape, MIC_pred.shape)
        return {'Feature1':eval_image(Image_CUT[1,...], MIC_pred, True),
                'Feature2':eval_image(Image_CUT[2,...], LNP_pred, True),
                'Feature3':eval_image(Image_CUT[3,...], ACT_pred, True)}
        
    if save_predictions:
        while out_path == None:
            out_path = askDIR()
        if filename == None:
            filename = os.path.basename(os.path.abspath(IMG_path))[:-4]

        np.save(file = f"{out_path}/{filename}.npy", arr=np.dstack([
            Image_CUT[1,...].astype(np.uint8), Image_CUT[2,...].astype(np.uint8), Image_CUT[3,...].astype(np.uint8), 
            ((MIC_pred/np.max(MIC_pred))*255).astype(np.uint8), 
            ((LNP_pred/np.max(LNP_pred))*255).astype(np.uint8), 
            ((ACT_pred/np.max(ACT_pred))*255).astype(np.uint8)
            ]))
        
    
def Segment_Image(IMG_path:str = None, 
                  patchsize:tuple = (256,256),
                  predictor = None, 
                  th_ER:float = 0.2, 
                  th_MIC:float = 0.2,
                  th_CLA:float = 0.2, 
                  device = torch.device("cuda"), 
                  lower_dyn_th:int = 0, 
                  upper_dyn_th:int = 1, 
                  activation = torch.nn.Identity(), 
                  test_zero:bool = False, 
                  R_DIL_PRE = None, 
                  kernel_size = (2,2),
                  iter:int = 1,
                  return_imgs = False,
                  revert:bool = False,
                  normalize:bool = False, 
                  standardize:bool = False,
                  image_overview:bool = True,
                  save_predictions:bool = False,
                  filename:str = None, 
                  out_path:str = None):

    def __normalize(arr):
        return arr/np.max(arr)
    
    if test_zero:
        IMG = np.zeros(shape = (2560,2560))
        
    else:       
        while not os.path.isfile(IMG_path):
            print('Please pick an image to segment!')
            IMG_path = askFILE()
             
        try:
            if IMG_path.endswith('.npy'):
                IMG = np.load(IMG_path) 
            elif IMG_path.endswith(('.png','.jpg','.jpeg','.tif','.tiff')):
                IMG = np.array(imread(IMG_path)).astype(np.uint16)
                if IMG.shape[2] == 3:
                    IMG = IMG[:,:,0]
        except:
            return print('Expected numpy .npy or image file format. Please check your image selection!')
    
    if (IMG.shape[0] % patchsize[0] == 0) & (IMG.shape[1] % patchsize[1] == 0):
        Image_CUT = IMG.copy()[...]
    
    elif (IMG.shape[0] % patchsize[0] == 0):
        Image_CUT = IMG.copy()
        Image_CUT = Image_CUT[:, :-(Image_CUT.shape[1] % patchsize[1])]
        
    elif (IMG.shape[1] % patchsize[1] == 0):
        Image_CUT = IMG.copy()
        Image_CUT = Image_CUT[:-(Image_CUT.shape[0] % patchsize[0]),:]
        
    else:
        Image_CUT = IMG.copy()
        Image_CUT = Image_CUT[:-(Image_CUT.shape[0] % patchsize[0]), :-(Image_CUT.shape[1] % patchsize[1])]
            
        
    
    NumImg_Y = int((np.floor(Image_CUT.shape[0]/patchsize[0])))
    NumImg_X = int((np.floor(Image_CUT.shape[1]/patchsize[1])))
    NumImg_tmp = NumImg_Y*NumImg_X
    
    print("Image has shape {} and is going to be cut into {} pieces of shape {}.".format(IMG.shape, int(NumImg_tmp),(patchsize)))
    
    Vanilla = deepcopy(Image_CUT)
    
    if ER_DIL_PRE == 'ER':
        import cv2 as cv
        Image_CUT = cv.erode(Image_CUT, np.ones(kernel_size, np.uint8), iterations=iter)
        
    elif ER_DIL_PRE == 'DIL':
        import cv2 as cv
        Image_CUT = cv.dilate(Image_CUT, np.ones(kernel_size, np.uint8), iterations=iter)
    
    Patches = patchify(Image_CUT, patchsize, patchsize[0])
    Data_In = np.reshape(Patches,(NumImg_tmp,1,patchsize[0],patchsize[1]))
    Data_In = Data_In.astype('float32')
    
    if not standardize:
        Data_In = Data_In/725#722#728#580#725#np.max(Data_In) #255

    predictor.eval()   
    Pred = np.zeros(shape=(NumImg_tmp,3, patchsize[0], patchsize[1]))
    
    for i,patch in enumerate(tqdm(Data_In, desc='Predicting...')):
        if not np.any(patch):
            Pred[i,...] = patch
            #print('SKIP')
            continue
        tmp = torch.from_numpy(patch[None,...].astype(np.float32))
        if standardize:
            tmp = ((tmp - tmp.mean())/tmp.std())
            tmp = torch.nan_to_num(tmp)
        tmp = tmp.to(device)    
        Pred[i,...] = activation(predictor(tmp)).cpu().detach().numpy()
    
    ####################################################################
    
    MIC = Pred[:,0,:,:]
    MIC = MIC - np.min(MIC.flatten())

    LNP = Pred[:,1,:,:]
    LNP = LNP - np.min(LNP.flatten())

    ACT = Pred[:,2,:,:]
    MIC = MIC - np.min(MIC.flatten())

    if not test_zero:
        dyn_th_pre_param = lower_dyn_th
        dyn_th_post_param = upper_dyn_th
        
        from filters import rank_filter
        
        tmp_mic, bord_mic = np.histogram(MIC, bins = 100)
        MIC_th = (bord_mic[np.argmax(tmp_mic)+dyn_th_pre_param], bord_mic[np.argmax(tmp_mic)+dyn_th_post_param])
        MIC[(MIC>=MIC_th[0]) & (MIC<=MIC_th[1])] = 0
        MIC[MIC < th_ER] = 0
            
        MIC_pred = unpatchify(np.reshape(MIC, (NumImg_Y,NumImg_X,patchsize[0],patchsize[1])), Image_CUT.shape)
        
        tmp_lnp, bord_lnp = np.histogram(LNP, bins = 100)
        LNP_th = (bord_lnp[np.argmax(tmp_lnp)+dyn_th_pre_param], bord_lnp[np.argmax(tmp_lnp)+dyn_th_post_param])
        LNP[(LNP>=LNP_th[0]) & (LNP<=LNP_th[1])] = 0
        LNP[LNP < th_MIC] = 0
        LNP_pred = unpatchify(np.reshape(LNP, (NumImg_Y,NumImg_X,patchsize[0],patchsize[1])), Image_CUT.shape)
        
        tmp_act, bord_act = np.histogram(ACT, bins = 100)
        ACT_th = (bord_act[np.argmax(tmp_act)+dyn_th_pre_param], bord_act[np.argmax(tmp_act)+dyn_th_post_param])
        ACT[(ACT>=ACT_th[0]) & (ACT<=ACT_th[1])] = 0
        ACT[ACT < th_CLA] = 0
        ACT_pred = unpatchify(np.reshape(ACT, (NumImg_Y,NumImg_X,patchsize[0],patchsize[1])), Image_CUT.shape)
        
    if ER_DIL_PRE == 'ER' and revert:
        MIC = cv.dilate(MIC, np.ones(kernel_size, np.uint8), iterations=iter)
        LNP = cv.dilate(LNP, np.ones(kernel_size, np.uint8), iterations=iter)
        ACT = cv.dilate(ACT, np.ones(kernel_size, np.uint8), iterations=iter)
        
        
    elif ER_DIL_PRE == 'DIL' and revert:
        MIC = cv.erode(MIC, np.ones(kernel_size, np.uint8), iterations=iter)
        LNP = cv.erode(LNP, np.ones(kernel_size, np.uint8), iterations=iter)
        ACT = cv.erode(ACT, np.ones(kernel_size, np.uint8), iterations=iter)
        
    if return_imgs:
        return MIC,LNP,ACT

    plt.style.use('dark_background')
    font = {'family' : 'Courier new',
            'weight' : 'bold',
            'size'   : 8}
    mpl.rc('font', **font)

    if save_predictions:
        while out_path == None:
            out_path = askDIR()
        if filename == None:
            filename = os.path.basename(os.path.abspath(IMG_path))[:-4]

        return np.save(file = f"{out_path}/{filename}.npy", arr=np.dstack([ 
            ((MIC_pred/np.max(MIC_pred))*255).astype(np.uint8), 
            ((LNP_pred/np.max(LNP_pred))*255).astype(np.uint8), 
            ((ACT_pred/np.max(ACT_pred))*255).astype(np.uint8)
            ]))

    if image_overview:
        fig, axs = plt.subplots(1,5,figsize = (20,5), dpi = 250)
        
        axs[0].set_title('Vanilla Image')
        img1 = axs[0].imshow(Vanilla, cmap = 'gray')
        fig.colorbar(img1, ax = axs[0])
        axs[0].axis('off')
        
        axs[1].set_title('Srg Overlay (ER+MIC+CLA) Diff')
        img2 = axs[1].imshow(__normalize(unpatchify(np.reshape(MIC+LNP+ACT, (NumImg_Y,NumImg_X,patchsize[0],patchsize[1])), Image_CUT.shape))  - __normalize(Vanilla), cmap = 'gray')
        fig.colorbar(img2, ax = axs[1])
        axs[1].axis('off')
        
        axs[2].set_title('Segmented ER')
        img3 = axs[2].imshow(unpatchify(np.reshape(MIC, (NumImg_Y,NumImg_X,patchsize[0],patchsize[1])), Image_CUT.shape), cmap = 'plasma')
        fig.colorbar(img3, ax = axs[2])
        axs[2].axis('off')
        
        axs[3].set_title('Segmented MIC')
        img4 = axs[3].imshow(unpatchify(np.reshape(LNP, (NumImg_Y,NumImg_X,patchsize[0],patchsize[1])), Image_CUT.shape), cmap = 'plasma')
        fig.colorbar(img4, ax = axs[3])
        axs[3].axis('off')
        
        axs[4].set_title('Segmented CLA')
        img5 = axs[4].imshow(unpatchify(np.reshape(ACT, (NumImg_Y,NumImg_X,patchsize[0],patchsize[1])), Image_CUT.shape), cmap = 'plasma')
        fig.colorbar(img5, ax = axs[4])
        axs[4].axis('off')

    elif not image_overview:
        fig, ((ax1,ax5),(ax2,ax6),(ax3,ax7),(ax4,ax8)) = plt.subplots(4,2,figsize = (15,10), dpi = 200)
            
        ax1.axis('off')
        ax1.set_title('Vanilla Image')
        im1 = ax1.imshow(Vanilla, cmap = 'gray')
        plt.colorbar(im1, ax = ax1)
        
        ax5.axis('off')
        ax5.set_title('Segmented Overlay')
        im5 = ax5.imshow(unpatchify(np.reshape(MIC+LNP+ACT, (NumImg_Y,NumImg_X,patchsize[0],patchsize[1])), Image_CUT.shape), cmap = 'gray')
        plt.colorbar(im5, ax = ax5)
        
        ax2.axis('off')
        ax2.set_title('Segmented ER')
        im2 = ax2.imshow(unpatchify(np.reshape(MIC, (NumImg_Y,NumImg_X,patchsize[0],patchsize[1])), Image_CUT.shape), cmap = 'plasma')
        plt.colorbar(im2, ax = ax2)
        
        ax3.axis('off')
        ax3.set_title('Segmented MIC')
        im3 = ax3.imshow(unpatchify(np.reshape(LNP, (NumImg_Y,NumImg_X,patchsize[0],patchsize[1])), Image_CUT.shape), cmap = 'plasma')
        plt.colorbar(im3, ax = ax3)
        
        ax4.axis('off')
        ax4.set_title('Segmented CLA')
        im4 = ax4.imshow(unpatchify(np.reshape(ACT, (NumImg_Y,NumImg_X,patchsize[0],patchsize[1])), Image_CUT.shape), cmap = 'plasma')
        plt.colorbar(im4, ax = ax4)
        
        if not test_zero:
            ax6.set_title('Seg ER Int. HIST')
            ax6.hist(MIC.flatten(), bins = 100, color='orange')
            
            ax7.set_title('Seg MIC Int. HIST')
            ax7.hist(LNP.flatten(), bins = 100, color='orange')
            
            ax8.set_title('Seg CLA Int. HIST')
            ax8.hist(ACT.flatten(), bins = 100, color='orange')