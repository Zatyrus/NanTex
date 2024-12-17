## Dependencies
import os
import torch
import datetime

# Typing
from typing import Dict, Any

## Custom dependencies
from NanTex_backend.deep_learning.model_src.Unet_setup_torch import UNet

from torch.utils.tensorboard import SummaryWriter

## Hyperparameters
hyperparameters:Dict[str,Any] = {
    "batch_size": 16,
    "epochs": 32,
    "learning_rate": 5e-4,
    "weight_decay": 0,
    "steps_per_epoch": 64,
    "val_per_epoch": 32
}

## Nework parameters
Unet_config:Dict[str,Any] = {
    "in_channels": 1,
    "num_fmaps": 16,
    "fmap_inc_factors": 3,
    "downsample_factors": [[2,2],[2,2],[2,2],[2,2]],
    "kernel_size_down": None,   # None means 3x3
    "kernel_size_up": None,     # None means 3x3
    "activation":"ReLU",        # Activation function
    "padding": "same",          # Padding mode
    "num_fmaps_out": None,      # None means in_channels
    "constant_upsample": False  # If True, use constant (probably bilinear) upsampling 
}

final_layer_config:Dict[str,Any] = {
    "in_channels": 16,
    "out_channels": 2,
    "kernel_size": 1,
    "padding": "same"
}

assembly_config:Dict[str,Any] = {
    "loss": torch.nn.MSELoss,
    "dtype": torch.FloatTensor,
    "activation": torch.nn.Identity,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "optimizer": torch.optim.Adam,
}

misc_config:Dict[str,Any] = {
    "logdir": "./logs",
    "model_name": "Unet",
}

## Assembly Line
# Main Model
Unet:torch.nn.Module
Unet = UNet(**Unet_config)

# Final Layer
final_layer:torch.nn.Module
final_layer = torch.nn.Conv2d(**final_layer_config)

# Assembly
model:torch.nn.Module
model = torch.nn.Sequential(Unet, final_layer)

# Activation
activation:torch.nn.Module
activation = assembly_config["activation"]()

# Loss
loss_fn:torch.nn.Module
loss_fn = assembly_config["loss"]()

# Optimizer
optimizer:torch.optim.Optimizer
optimizer = assembly_config["optimizer"](model.parameters(),
                                         lr = hyperparameters["learning_rate"],
                                         weight_decay = hyperparameters["weight_decay"])

## Device
device:torch.device
device = assembly_config["device"]

# Transfer components to device
model = model.to(device)
activation = activation.to(device)
loss_fn = loss_fn.to(device)

## Logging
# Tensorboard
writer:SummaryWriter
writer = SummaryWriter(log_dir = os.path.join(misc_config["logdir"], 
                                              datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

## Complete Model
assembled_model:Dict[str,Any] = {
    "model": model,
    "activation": activation,
    "loss_fn": loss_fn,
    "optimizer": optimizer,
    "device": device,
    "writer": writer
}

## Namespace
__all__ = ["assembled_model", "hyperparameters", "Unet_config", "final_layer_config", "assembly_config", "misc_config"]
