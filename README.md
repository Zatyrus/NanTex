# NanTex
 Nanotextural analysis of SMLM data

## Installation Guide
1. Install [Python Poetry](https://python-poetry.org/docs/).
2. Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install). Create a new conda environent and activate it.
```bash
conda create -n NanTex python==3.11
conda activate NanTex
```
3. Navigate to the directory where you have downloaded the NanTex repository.
```bash
cd /path/to/NanTex
```
4. Run the following command with your activated virtual environent to install all the required packages:
```bash
poetry install
```
5. After that, you are ready to use NanTex. You can run the Jupyter notebooks provided in the `notebooks` folder to get started. This installation retuires an active internet connection to download the necessary packages. It should take around 10-15 minutes depending on your internet speed.

**NOTE**: The deep-learning implementation used in this package is based on PyTorch. Based on your available hardware and operating system, you may have to install and or update your PyTorch and CUDA toolkit manually. Follow the steps below if you have a compatible NVIDIA GPU.

1. Update your GPU Drivers and [CUDA Toolkit](https://developer.nvidia.com/cuda/toolkit) to the latest version. You can find the latest version supported by PyTorch [here](https://pytorch.org/get-started/locally/).
2. Activate the environment and pull the latest version of torch. You will find the pip command on the PyTorch website (link above).

## System Requirements

### Software Dependencies

**Operating System:** \
Windows 10 (22H2) \
LINUX (some UX/UI methods might not work as intended)

**Python** \
python = ">=3.11,<3.14" \
numpy = "^2.2.0" \
matplotlib = "^3.10.0" \
tqdm = "^4.67.1" \
scikit-image = "^0.25.0" \
jupyter = "^1.1.1" \
ipykernel = "^6.29.5" \
ray = {extras = ["default"], version = "^2.49.1"} \
ezray = "^1.1.4" \
pytorch-msssim = "^1.0.0" \
albumentations = "^2.0.8" \
torch = {version = "^2.8.0+cu129"}