import sys, comet_ml
from data import datamodule
from train import fit
import torch
import os
#from private.comet_key import key
import torchmetrics
from ray import tune


def config_dict (arch,optim):
     ######## Config for architectures ############
    if optim=="pbt":
       from config_pbt import config_inc_pbt,config_res_pbt,config_alex_pbt,config_vgg_pbt,scheduler_p
       if arch=="inc":
          config=config_inc_pbt
       if arch=="res":
          config=config_res_pbt
       if arch=="alex":
          config=config_alex_pbt
       if arch=="vgg":
          config=config_vgg_pbt
    else:
       from config_asha import config_inc,config_res,config_alex,config_vgg,scheduler_a
       if arch=="inc":
          config=config_inc
       if arch=="res":
          config=config_res
       if arch=="alex":
          config=config_alex
       if arch=="vgg":
          config=config_vgg
    return config

def main (num_samples=40, num_epochs=50, folder="Dataset", arch='inc',optim=None):
    os.environ["SLURM_JOB_NAME"] = "bash"
    data_dir = os.path.join(os.getcwd(), folder)
    ###### scheduler switcher ##########
    if optim=="pbt":
          from config_pbt import scheduler_p
    else: 
          from config_asha import scheduler_a
    scheduler_switch={
        "asha":scheduler_a,
        "pbt":scheduler_p
    }
    ######### tune.with_parameters inception net #######
    trainable = tune.with_parameters(
        fit.train_fn,
        model_arch=arch,
        data_dir=data_dir,
        num_epochs=num_epochs,
        num_gpus=1)
    analysis = tune.run(
        trainable,
        resources_per_trial={
            "cpu": 16,
            "gpu": 1
        },
        local_dir="../tune-2-analyses/results",
        #verbose=2,
        config=config_dict(arch,optim),
        scheduler=scheduler_switch[optim],# if optim,
        #metric="loss" ,
        #mode="min" ,
        #progress_reporter=reporter_inc,
        num_samples=num_samples,
        name="tune-"+arch+"-"+optim)

    print(analysis.best_config)


if __name__ == "__main__":
    main(num_samples=100, num_epochs=35, folder="Sort-Dataset", arch='res', optim="asha")
    #main(num_samples=100, num_epochs=35, folder="Sort-Dataset", arch='alex', optim='asha')
    #main(num_samples=40, num_epochs=35, folder="Dataset", arch='alex', opt='pbt')
    #main(num_samples=40, num_epochs=35, folder="Dataset", arch='vgg', opt='pbt')
