# NanTex
 Nanotextural analysis of SMLM data

## Installation Guide
1. Install [Python Poetry](https://python-poetry.org/docs/).
2. (Optional) Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install). Create a new conda environent and activate it.
```bash
conda create -n NanTex python==3.11
conda actiavte NanTex
```
3. Navigate to the directory where you have downloaded the NanTex repository.
4. Run the following command to create a new virtual environent with all the required packages:
```bash
poetry install
```
NOTE: The deep-learning implementation used in this package is based on PyTorch.
Based on your available hardware and operating system, you may have to install

1. Update your GPU Drivers and CUDA Toolkit to the latest version. You can find the latest version supported by PyTorch [here](https://pytorch.org/get-started/locally/).
2. Activate the dev environment and pull the latest version of torch. You will find the pip command on the PyTorch website (link above).

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