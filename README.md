# NanTex
 Nanotextural analysis of SMLM data

## Setup Guide
1. Install Anaconda/Miniconda (just any Cconda distribution).
2. Open Anaconda Prompt and navigate to the directory where you have downloaded the NanTex repository.
3. Run the following command to create two new conda environments with all the required packages:
```bash
conda env create -f NanTex_dev.yml
conda env create -f NanTex_eval.yml
```
NOTE: The above command will create two new conda environments named `NanTex_dev` and `NanTex_eval`, which are needed due to version conflicts. The first environment is for development purposes and the second one is for evaluation purposes. You can activate the environment by running the following command:
```bash
conda activate NanTex_dev/Eval
```
4. Update your GPU Drivers and CUDA Toolkit to the latest version. You can find the latest version supported by PyTorch [here](https://pytorch.org/get-started/locally/).
5. Activate the dev environment and pull the latest version of torch. You will find the pip command on the PyTorch website (link above).