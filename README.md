# Description
This codes inlcude a training function to tune a Convolutional neural network model using ray tune implementation of the ASHA scheduler and Population Based Training scheduler.

# Links
A primary reference for ASHA is here:  https://arxiv.org/pdf/1810.05934.pdf
A good resource on PBT is here:  https://deepmind.com/blog/article/population-based-training-neural-networks

At the time of this commit, the address: https://docs.ray.io/en/latest/tune/api_docs/schedulers.html hosts some more information on the ASHA and Population Based Training schedulers implementation on ray 

There are two "main" functions: one that implements the ASHA Schduler - main_asha.py, and another main_pbt.py that implements the PBT scheduler 

The Dataset should be located in a folder and its path properly identified as an input to the main function

The models to be optimized should be accordingly made configurable. The data loaders should also be in the same fashion. These codes were tested with models and dataloaders in pytorchlightning.

I plan to include the models and dataloader in a future commit.

# Dependencies
The code is written in python, and uses pytorch, pytorchlightning and ray tune.  
