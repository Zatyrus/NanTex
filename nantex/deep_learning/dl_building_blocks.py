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
except Exception as e:
    print(f"Error occurred while importing tqdm: {e}")
    from tqdm import tqdm

## Additional Metrices
from pytorch_msssim import SSIM, MS_SSIM
from torch.nn import MSELoss


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


def normalize_tensor_01(
    tensor: torch.Tensor, min_val: float, max_val: float
) -> torch.Tensor:
    return (tensor - min_val) / (
        max_val - min_val + 1e-12
    )  # add 1e-12 to avoid negative values and division by zero


def denormalize_tensor_01(tensor: torch.Tensor, data_range: float) -> torch.Tensor:
    return tensor * data_range


def normalize_tensor_11(
    tensor: torch.Tensor, min_val: float, max_val: float
) -> torch.Tensor:
    return (tensor - min_val) / (max_val - min_val) * 2 - 1


def denormalize_tensor_11(
    tensor: torch.Tensor, min_val: float, max_val: float
) -> torch.Tensor:
    return (tensor + 1) / 2 * (max_val - min_val) + min_val


# %% Main Training Loop
def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    net: torch.nn.Sequential,
    loss_fn: torch.nn.Module,
    activation: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    steps_per_epoch: int,
    val_per_epoch: int,
    save_dir: str = "./model",
    num_channels: int = 3,
    data_range: float = 1.0,
    write_val_per_feature: bool = False,
    write_SSIM_MSSSIM_on_the_fly: bool = False,
    uses_11_normalization: bool = False,
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
    epoch_step_counter: int = 0
    optimal_loss = np.inf
    optimal_msssim = 0
    if write_val_per_feature:
        optimal_feature_loss = {i: np.inf for i in range(num_channels)}
        optimal_feature_msssim = {i: 0 for i in range(num_channels)}

    # initialize experimental metrics
    MSE_val_Metr: MSELoss

    SSIM_Metr: SSIM
    MSSSIM_Metr: MS_SSIM

    SSIM_Metr = SSIM(
        data_range=data_range,
        size_average=True,
        channel=num_channels,
        nonnegative_ssim=True,
    )
    MSSSIM_Metr = MS_SSIM(
        data_range=data_range, size_average=True, channel=num_channels
    )

    if write_val_per_feature:
        MSE_val_Metr: MSELoss
        SSIM_val_Metr: SSIM
        MSSSIM_val_Metr: MS_SSIM

        MSE_val_Metr = MSELoss()
        SSIM_val_Metr = SSIM(
            data_range=data_range, size_average=True, channel=1, nonnegative_ssim=True
        )
        MSSSIM_val_Metr = MS_SSIM(data_range=data_range, size_average=True, channel=1)

    # send to device
    SSIM_Metr.to(device)
    MSSSIM_Metr.to(device)
    if write_val_per_feature:
        MSE_val_Metr.to(device)
        SSIM_val_Metr.to(device)
        MSSSIM_val_Metr.to(device)

    # Grab a training batch
    tmp_loader = iter(train_loader)

    # Grab a validation batch
    tmp_val_loader = iter(val_loader)

    # update steps per epoch based on dataset size
    steps_per_epoch = min(steps_per_epoch, len(train_loader))
    val_per_epoch = min(val_per_epoch, len(val_loader))

    # setup

    # talk to the user
    print(f"Training for {epochs} epochs, {steps_per_epoch} steps per epoch.")
    print(f"Validating every epoch, {val_per_epoch} steps per validation.")

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
                # setup epoch counters
                train_count = 0
                val_count = 0

                ## Batch loop
                for feature, label in tmp_loader:
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

                    # denormalize if needed
                    if uses_11_normalization:
                        label = denormalize_tensor_11(label, 0.0, data_range)
                        pred = denormalize_tensor_11(pred, 0.0, data_range)

                    # Write to tensorboard
                    writer.add_scalar(
                        tag="train/MSE",
                        scalar_value=loss_value.cpu().detach().numpy(),
                        global_step=epoch_step_counter,
                    )

                    # extended on the fly evaluation metrics
                    if write_SSIM_MSSSIM_on_the_fly:
                        writer.add_scalar(
                            tag="train/SSIM",
                            scalar_value=SSIM_Metr(pred, label).cpu().detach().numpy(),
                            global_step=epoch_step_counter,
                        )
                        writer.add_scalar(
                            tag="train/MSSSIM",
                            scalar_value=MSSSIM_Metr(pred, label)
                            .cpu()
                            .detach()
                            .numpy(),
                            global_step=epoch_step_counter,
                        )

                    # Handle progress
                    pbar.set_description(
                        f"Current batch loss: {loss_value.cpu().detach().numpy():.3e}"
                    )
                    # update batch progress
                    batch_pbar.update(1)
                    # write epoch step counter
                    epoch_step_counter += 1

                    # break condition for steps per epoch
                    train_count += 1
                    if train_count >= steps_per_epoch:
                        break

                # format batch_pbar
                batch_pbar.set_description("Checkpoint reached ...")

                # create checkpoints #
                torch.save(
                    net.state_dict(), f"{checkpoint_path}/checkpoint_epoch_{epoch}.pt"
                )

                # %%  #### Validation #### %% # <- AFTER EACH EPOCH

                # reset batch_pbar
                batch_pbar.reset(total=val_per_epoch)
                batch_pbar.set_description("Currently validating...")
                batch_pbar.colour = "orange"

                # set model to eval mode
                net.eval()

                # per default, we write validation metrices as averages
                if not write_val_per_feature:
                    # initialize accumulators
                    acc_loss = []
                    acc_ssim = []
                    acc_msssim = []

                    for feature, label in tmp_val_loader:
                        ## MOVE TO DEVICE
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

                        # denormalize if needed
                        if uses_11_normalization:
                            label = denormalize_tensor_11(label, 0.0, data_range)
                            val_pred = denormalize_tensor_11(val_pred, 0.0, data_range)

                        acc_loss.append(loss_value.cpu().detach().numpy())
                        acc_ssim.append(
                            SSIM_Metr(val_pred, label).cpu().detach().numpy()
                        )
                        acc_msssim.append(
                            MSSSIM_Metr(val_pred, label).cpu().detach().numpy()
                        )

                        # update batch_pbar
                        batch_pbar.update(1)

                        # break condition for val per epoch
                        val_count += 1
                        if val_count >= val_per_epoch:
                            break

                # we can also write them per feature
                if write_val_per_feature:
                    acc_loss = []
                    acc_ssim = []
                    acc_msssim = []

                    # per feature metrics
                    collector = {"val_MSE": {}, "val_SSIM": {}, "val_MSSSIM": {}}
                    for i in range(num_channels):
                        collector["val_MSE"][i] = []
                        collector["val_SSIM"][i] = []
                        collector["val_MSSSIM"][i] = []

                    for feature, label in tmp_val_loader:
                        ## MOVE TO DEVICE
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
                        # append MSE
                        acc_loss.append(loss_value.cpu().detach().numpy())
                        for i in range(num_channels):
                            collector["val_MSE"][i].append(
                                MSE_val_Metr(
                                    val_pred[:, i, :, :][:, np.newaxis, :, :],
                                    label[:, i, :, :][:, np.newaxis, :, :],
                                )
                                .cpu()
                                .detach()
                                .numpy()
                            )

                        # denormalize if needed
                        if uses_11_normalization:
                            label = denormalize_tensor_11(label, 0.0, data_range)
                            val_pred = denormalize_tensor_11(val_pred, 0.0, data_range)

                        acc_ssim.append(
                            SSIM_Metr(val_pred, label).cpu().detach().numpy()
                        )
                        acc_msssim.append(
                            MSSSIM_Metr(val_pred, label).cpu().detach().numpy()
                        )
                        for i in range(num_channels):
                            collector["val_SSIM"][i].append(
                                SSIM_val_Metr(
                                    val_pred[:, i, :, :][:, np.newaxis, :, :],
                                    label[:, i, :, :][:, np.newaxis, :, :],
                                )
                                .cpu()
                                .detach()
                                .numpy()
                            )
                            collector["val_MSSSIM"][i].append(
                                MSSSIM_val_Metr(
                                    val_pred[:, i, :, :][:, np.newaxis, :, :],
                                    label[:, i, :, :][:, np.newaxis, :, :],
                                )
                                .cpu()
                                .detach()
                                .numpy()
                            )

                        # update batch_pbar
                        batch_pbar.update(1)

                        # break condition for val per epoch
                        val_count += 1
                        if val_count >= val_per_epoch:
                            break

                    # reset batch_pbar
                    batch_pbar.colour = "green"
                    batch_pbar.set_description("Writing to tensorboard...")

                    # write to tensorboard
                    writer.add_scalar(
                        tag="val/MSE", scalar_value=np.mean(acc_loss), global_step=epoch
                    )
                    writer.add_scalar(
                        tag="val/SSIM",
                        scalar_value=np.mean(acc_ssim),
                        global_step=epoch,
                    )
                    writer.add_scalar(
                        tag="val/MSSSIM",
                        scalar_value=np.mean(acc_msssim),
                        global_step=epoch,
                    )

                    # write feature specific
                    for i in range(num_channels):
                        writer.add_scalar(
                            tag=f"val/MSE/channel_{i}",
                            scalar_value=np.mean(collector["val_MSE"][i]),
                            global_step=epoch,
                        )
                        writer.add_scalar(
                            tag=f"val/SSIM/channel_{i}",
                            scalar_value=np.mean(collector["val_SSIM"][i]),
                            global_step=epoch,
                        )
                        writer.add_scalar(
                            tag=f"val/MSSSIM/channel_{i}",
                            scalar_value=np.mean(collector["val_MSSSIM"][i]),
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
                if write_val_per_feature:
                    for i in range(num_channels):
                        if np.mean(collector["val_MSE"][i]) < optimal_feature_loss[i]:
                            optimal_feature_loss[i] = np.mean(collector["val_MSE"][i])
                            torch.save(
                                net.state_dict(),
                                f"{checkpoint_path}/model_best_channel_{i}.pt",
                            )

                        if (
                            np.mean(collector["val_MSSSIM"][i])
                            > optimal_feature_msssim[i]
                        ):
                            optimal_feature_msssim[i] = np.mean(
                                collector["val_MSSSIM"][i]
                            )
                            torch.save(
                                net.state_dict(),
                                f"{checkpoint_path}/model_optimal_msssim_channel_{i}.pt",
                            )

                # Reset batch_pbar to training mode
                batch_pbar.reset(total=steps_per_epoch)
                batch_pbar.set_description("Processing batch...")
                batch_pbar.colour = "dodgerblue"

                # generate new val loader iterator
                tmp_val_loader = iter(val_loader)

                # generate new train loader iterator
                tmp_loader = iter(train_loader)

                # Reset model to training mode
                net.train()

                # update progress bar
                pbar.update(1)

    ## Save final model
    torch.save(net.state_dict(), f"{checkpoint_path}/model_final.pt")


# %% Validation
def validate(Any) -> Any:
    pass
