## Dependencies
import os
import pathlib
import datetime
import torch
import numpy as np
# from sewar.full_ref import rmse, uqi, ergas, rase, sam, vifp

# Typing
from torch.utils.data import DataLoader
from typing import Tuple, Any, NoReturn, Callable

# for progress bar
# detect jupyter notebook
from IPython import get_ipython

try:
    ipy_str = str(type(get_ipython()))
    if "zmqshell" in ipy_str:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except:
    from tqdm import tqdm

## Custom dependencies
from NanTex_backend.deep_learning.train_loss_experimental.ms_ssim_loss import (
    SSIM,
    MSSSIM,
)


from torch.utils.tensorboard import SummaryWriter

## Setup building blocks for deep learning


# %% Model epoch Definition
def model_step(
    model: torch.nn.Module,
    loss_fn: Callable,
    optimizer: Callable,
    feature: torch.Tensor,
    label: torch.Tensor,
    activation: Callable,
    is_training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # zero gradients if training
    if is_training:
        optimizer.zero_grad()

    # forward
    logits = model(feature)

    # final activation
    predicted = activation(logits)

    # pass through loss
    loss_value = loss_fn(input=predicted, target=label)

    # backward if training mode
    if is_training:
        loss_value.backward()
        optimizer.step()

    return loss_value, predicted


def prepare_routine(
    log_path: str, checkpoint_path: str, *args, **kwargs
) -> Tuple[str, str]:
    ## append datetime to log path and checkpoint path
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = f"{log_path}/{now}/logs"
    checkpoint_path = f"{checkpoint_path}/{now}/checkpoints"

    # check for log directory
    if not os.path.exists(log_path):
        pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)

    # check for checkpoint directory
    if not os.path.exists(checkpoint_path):
        pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    return log_path, checkpoint_path


# %% Main Training Loop
def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    net: torch.nn.Sequential,
    loss_fn: torch.nn.Module,
    activation: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dtype: torch.dtype,
    device: torch.device,
    epochs: int,
    steps_per_epoch: int,
    val_per_epoch: int,
    save_dir: str = "./model",
    batchsize: int = 16,
) -> NoReturn:
    # prepare routine
    log_path, checkpoint_path = prepare_routine(
        log_path=save_dir, checkpoint_path=save_dir
    )

    writer: SummaryWriter
    writer = SummaryWriter(log_dir=log_path)

    # set train flags, initialize step
    net.train()
    loss_fn.train()

    # initialize epoch and loss
    epoch = 0
    print_loss = 0
    optimal_loss = np.inf
    optimal_msssim = 0

    # initialize experimental metrics
    SSIM_Metr: SSIM
    MSSSIM_Metr: MSSSIM

    SSIM_Metr = SSIM()
    MSSSIM_Metr = MSSSIM()

    SSIM_Metr.to(device)
    MSSSIM_Metr.to(device)

    # Grab a training batch
    tmp_loader = iter(train_loader)

    # Grab a validation batch
    tmp_val_loader = iter(val_loader)

    with tqdm(
        total=epochs,
        leave=True,
        desc="Current batch loss: 0",
        position=0,
        colour="deeppink",
    ) as pbar:
        with tqdm(
            total=steps_per_epoch,
            leave=True,
            desc="Processing batch...",
            position=1,
            colour="dodgerblue",
        ) as batch_pbar:
            # Global training loop
            for epoch in range(epochs):
                ## Batch loop
                for _ in range(steps_per_epoch):
                    # Grab a training batch
                    feature, label = next(tmp_loader)

                    ## MOVE TO DEVICE
                    # absolutely crucial
                    label: torch.Tensor
                    feature: torch.Tensor
                    label = label.to(device)
                    feature = feature.to(device)

                    ## Progress through model
                    loss_value, pred = model_step(
                        model=net,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        feature=feature,
                        label=label,
                        activation=activation,
                        is_training=True,
                    )

                    # Write to tensorboard
                    writer.add_scalar(
                        tag="MSE",
                        scalar_value=loss_value.cpu().detach().numpy(),
                        global_step=epoch,
                    )
                    writer.add_scalar(
                        tag="SSIM",
                        scalar_value=SSIM_Metr(pred, label).cpu().detach().numpy(),
                        global_step=epoch,
                    )
                    writer.add_scalar(
                        tag="MSSSIM",
                        scalar_value=MSSSIM_Metr(pred, label).cpu().detach().numpy(),
                        global_step=epoch,
                    )

                    # Handle progress
                    pbar.set_description(
                        f"Current batch loss: {loss_value.cpu().detach().numpy():.3e}"
                    )
                    batch_pbar.update(1)

                # format batch_pbar
                batch_pbar.set_description("Checkpoint reached ...")

                # create checkpoints #
                torch.save(
                    net.state_dict(), f"{checkpoint_path}/checkpoint_epoch_{epoch}.pt"
                )

                #### Validation ####
                # reset batch_pbar
                batch_pbar.reset(total=val_per_epoch)
                batch_pbar.set_description("Currently validating...")
                batch_pbar.colour = "orange"

                # set model to eval mode
                net.eval()

                # initialize accumulators
                acc_loss = []
                acc_ssim = []
                acc_msssim = []

                for _ in range(val_per_epoch):
                    feature, label = next(tmp_val_loader)

                    ## MOVE TO DEVICE
                    # absolutely crucial
                    label: torch.Tensor
                    feature: torch.Tensor
                    feature = feature.to(device)
                    label = label.to(device)

                    # progress through model
                    loss_value, val_pred = model_step(
                        model=net,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        feature=feature,
                        label=label,
                        activation=activation,
                        is_training=False,
                    )

                    acc_loss.append(loss_value.cpu().detach().numpy())
                    acc_ssim.append(SSIM_Metr(val_pred, label).cpu().detach().numpy())
                    acc_msssim.append(
                        MSSSIM_Metr(val_pred, label).cpu().detach().numpy()
                    )

                    ## ON THE FLY VALIDATION HOOK ##

                    # update batch_pbar
                    batch_pbar.update(1)

                # reset batch_pbar
                batch_pbar.colour = "green"
                batch_pbar.set_description("Writing to tensorboard...")

                # write to tensorboard
                writer.add_scalar(
                    tag="val_MSE", scalar_value=np.mean(acc_loss), global_step=epoch
                )
                writer.add_scalar(
                    tag="val_SSIM", scalar_value=np.mean(acc_ssim), global_step=epoch
                )
                writer.add_scalar(
                    tag="val_MSSSIM",
                    scalar_value=np.mean(acc_msssim),
                    global_step=epoch,
                )

                # reset batch_pbar
                batch_pbar.colour = "green"
                batch_pbar.set_description("Writing to checkpoints...")

                # write weight checkpoints
                if np.mean(acc_loss) < optimal_loss:
                    optimal_loss = np.mean(acc_loss)
                    torch.save(net.state_dict(), f"{checkpoint_path}/model_best.pt")

                if np.mean(acc_msssim) > optimal_msssim:
                    optimal_msssim = np.mean(acc_msssim)
                    torch.save(
                        net.state_dict(), f"{checkpoint_path}/model_optimal_msssim.pt"
                    )

                # Reset batch_pbar to training mode
                batch_pbar.reset(total=steps_per_epoch)
                batch_pbar.set_description("Processing batch...")
                batch_pbar.colour = "dodgerblue"

                # Reset model to training mode
                net.train()

                # update progress bar
                pbar.update(1)

    ## Save final model
    torch.save(net.state_dict(), f"{checkpoint_path}/model_final.pt")


# %% Validation
def validate(Any) -> Any:
    pass
