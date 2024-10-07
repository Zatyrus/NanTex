## Dependencies
import os, sys
import datetime, time
import torch
import numpy as np
from sewar.full_ref import rmse, uqi, ergas, rase, sam, vifp

# Typing
from torch.utils.data import DataLoader
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

## Custom dependencies
from NanTex_backend.deep_learning.train_loss_experimental.ms_ssim_loss import SSIM, MSSSIM
from NanTex_backend.Util.bit_generator_utils import default_rng, initialize_generator, spawn_generator


from torch.utils.tensorboard import SummaryWriter

## Setup building blocks for deep learning

#%% Model Step Definition
def model_step(model:torch.nn.Module, 
               loss_fn:Callable, 
               optimizer:Callable, 
               feature:torch.Tensor, 
               label:torch.Tensor, 
               activation:Callable,
               is_training:bool = True
               )->Tuple[torch.Tensor, torch.Tensor]:
    
    # zero gradients if training
    if is_training:
        optimizer.zero_grad()
    
    # forward
    logits = model(feature)
        
    # final activation
    predicted = activation(logits)

    # pass through loss
    loss_value = loss_fn(input = predicted, 
                         target = label)
 
    # backward if training mode
    if is_training:
        loss_value.backward()
        optimizer.step()
    
    return loss_value, predicted

#%% Main Training Loop
def train(train_loader:DataLoader, 
          val_loader:DataLoader, 
          net:torch.nn.Sequential, 
          loss_fn:torch.nn.Module, 
          activation:torch.nn.Module, 
          optimizer:torch.optim.Optimizer, 
          dtype:torch.dtype, 
          writer:SummaryWriter, 
          device:torch.device, 
          training_steps:int, 
          log_interval:int = 100,
          save_interval:int = 500, 
          save_dir:str = './model'
          )-> NoReturn:

    # set train flags, initialize step
    net.train() 
    loss_fn.train()
    
    # initialize step and loss
    step = 0
    print_loss = 0
    optimal_loss = np.infty
    optimal_msssim = 0
    
    # initialize experimental metrics
    SSIM_Metr:SSIM
    MSSSIM_Metr:MSSSIM
    
    SSIM_Metr = SSIM()
    MSSSIM_Metr = MSSSIM()
    
    SSIM_Metr.to(device)
    MSSSIM_Metr.to(device)

    with tqdm(total = training_steps,
              leave=False, 
              desc='Current batch loss: 0') as pbar:
        
        while step < training_steps:
            
            # Grab a training batch
            tmp_loader = iter(train_loader)
            
            ## Training
            for feature, label in tmp_loader:
                
                ## MOVE TO DEVICE
                # absolutely crucial
                label:torch.Tensor
                feature:torch.Tensor
                label = label.to(device)
                feature = feature.to(device)
                
                ## Progress through model
                loss_value, pred = model_step(model = net,
                                              loss_fn = loss_fn,
                                              optimizer = optimizer,
                                              feature = feature,
                                              label = label,
                                              activation = activation,
                                              is_training = True)
                
                # Write to tensorboard
                writer.add_scalar(tag = 'MSE', 
                                  scalar_value = loss_value.cpu().detach().numpy(),
                                  global_step = step)
                writer.add_scalar(tag = 'SSIM',
                                  scalar_value = SSIM_Metr(pred,label).cpu().detach().numpy(),
                                  global_step = step)
                writer.add_scalar(tag = 'MSSSIM', 
                                  scalar_value = MSSSIM_Metr(pred,label).cpu().detach().numpy(), 
                                  global_step = step)
                
                # Handle progress
                pbar.set_description(f"Current batch loss: {loss_value.cpu().detach().numpy():.3e}") 
                pbar.update(1)
                
                # Update step
                step += 1
                
                # create checkpoints
                if step % save_interval == 0:
                    torch.save(net.state_dict(), f"{save_dir}/checkpoints/checkpoint_{step}.pt")
                
                ## Validation
                if step % log_interval == 0:
                    # Grab a validation batch
                    tmp_val_loader = iter(val_loader)
                    
                    # set model to eval mode
                    net.eval()

                    # initialize accumulators
                    acc_loss = []
                    acc_ssim = []
                    acc_msssim = []
                    condition = 0
                    
                    for feature, label in tmp_val_loader:   
                        pbar.set_description("Currently validating: {}".format(condition))   
                        
                        
                        ## MOVE TO DEVICE
                        # absolutely crucial
                        label:torch.Tensor
                        feature:torch.Tensor    
                        feature = feature.to(device)
                        label = label.to(device)

                        
                        # progress through model
                        loss_value, val_pred = model_step(model=net, 
                                                          loss_fn=loss_fn, 
                                                          optimizer=optimizer, 
                                                          feature=feature, 
                                                          label=label, 
                                                          activation=activation, 
                                                          is_training = False)
                        
                        
                        acc_loss.append(loss_value.cpu().detach().numpy())
                        acc_ssim.append(SSIM_Metr(val_pred,label).cpu().detach().numpy())
                        acc_msssim.append(MSSSIM_Metr(val_pred,label).cpu().detach().numpy())
                        
                        # if condition == 0:
                        #     l = validate(label.cpu().detach().numpy(), val_pred.cpu().detach().numpy(), l = None)
                        # else:
                        #     l = validate(label.cpu().detach().numpy(), val_pred.cpu().detach().numpy(), l = l)
                        
                        # update condition
                        condition += 1
                        
                        # if condition == 5:
                        #     break
    
                    # write to tensorboard
                    writer.add_scalar(tag = "val_MSE", 
                                      scalar_value = np.mean(acc_loss), 
                                      global_step = step)
                    writer.add_scalar(tag = 'val_SSIM',
                                      scalar_value = np.mean(acc_ssim),
                                      global_step = step)
                    writer.add_scalar(tag = 'val_MSSSIM',
                                      scalar_value = np.mean(acc_msssim), 
                                      global_step = step)
                                           
                    # write weight checkpoints
                    if np.mean(acc_loss) < optimal_loss:
                        optimal_loss = np.mean(acc_loss)
                        torch.save(net.state_dict(), f'{save_dir}/checkpoints/model_best.pt')
                        
                    if np.mean(acc_msssim) > optimal_msssim:
                        optimal_msssim = np.mean(acc_msssim)
                        torch.save(net.state_dict(), f'{save_dir}/checkpoints/model_optimal_msssim.pt')
                    
                    torch.save(net.state_dict(), f'{save_dir}/checkpoints/model_current.pt')
                    
                    # Reset model to training mode
                    net.train()
                    
    ## Save final model
    torch.save(net.state_dict(), f'{save_dir}/model_final.pt')

#%% Validation
def validate(feature:np.ndarray, 
             pred:np.ndarray, 
             val_dict:Dict[str,List[float]] = None)->Dict[str,List[float]]:
    if val_dict == None:
        val_dict = {f"feature_{i}":{key:[] for key in ['RMSE','UQI','ERGAS','RASE','SAM','VIF']} for i in range(feature.shape[1])}
    for i in range(feature.shape[0]):
        for j in range(feature.shape[1]):
            for key, val in zip(val_dict.keys(),[rmse, uqi, ergas, rase, sam, vifp]):
                val_dict[f"feature_{i}"][key].append(val(pred[i,j,...],feature[i,j,...]))
            # l[j][0].append(rmse(pred[i,j,...],feature[i,j,...]))
            # l[j][1].append(uqi(pred[i,j,...],feature[i,j,...]))
            # l[j][2].append(ergas(pred[i,j,...],feature[i,j,...]))
            # l[j][3].append(rase(pred[i,j,...],feature[i,j,...]))
            # l[j][4].append(sam(pred[i,j,...],feature[i,j,...]))
            # l[j][5].append(vifp(pred[i,j,...],feature[i,j,...]))
    return val_dict
    
def write(val_dict, 
          writer, 
          step):
    for feature, val_subdict in val_dict.items():
        for key, val in val_subdict.items():
            writer.add_scalar(f"{feature}_{key}", np.mean(val), global_step = step)
    
    
    # for feature_count in range(len(val_dict.keys())):
    #     writer.add_scalar(f'{desc}_FT_{k}_RMSE',np.mean(val_list[k][0]), global_step=step)
    #     writer.add_scalar(f'{desc}_FT_{k}_UQI',np.mean(val_list[k][1]), global_step=step)
    #     writer.add_scalar(f'{desc}_FT_{k}_ERGAS',np.mean(val_list[k][2]), global_step=step)
    #     writer.add_scalar(f'{desc}_FT_{k}_RASE',np.mean(val_list[k][3]), global_step=step)
    #     writer.add_scalar(f'{desc}_FT_{k}_SAM',np.mean(val_list[k][4]), global_step=step)
    #     writer.add_scalar(f'{desc}_FT_{k}_VIF',np.mean(val_list[k][5]), global_step=step)