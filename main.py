import sys, comet_ml
from data import datamodule
from train import fit
import torch
import os
#from private.comet_key import key
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
import torchmetrics
from ray import tune
from config import *


def main (num_samples=40, num_epochs=50, folder="Dataset", arch='inc',optim=None):
    os.environ["SLURM_JOB_NAME"] = "bash"
    data_dir = os.path.join(os.getcwd(), folder)


    ######## ASHA Scheduler #################
    scheduler_a = ASHAScheduler(
        max_t=num_epochs,
        metric="loss",
        mode="min",
        grace_period=1,
        reduction_factor=2)
    ######## PBT Scheduler ##################
    scheduler_p = PopulationBasedTraining(
        perturbation_interval=4,
        hyperparam_mutations={
            "lr": tune.loguniform(1e-4, 1e-1),
            "mm":[0.6,0.9,1.2],
            "dp":[0,0.9,0.995],
            "wD":[0.000008,0.00001,0.00003 ],
            "batch_size": [32, 48, 96]
        },
        metric="loss",
        mode="min"
        )
    ###### scheduler switcher ##########
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

def config_dict (arch,optim):
     ######## Config for architectures ############
    if arch =='inc':
       config=config_inc_pbt if optim=="pbt" else config_inc
    elif arch =='res':
       config =config_res_pbt if optim=="pbt" else config_res
    elif arch == 'alex':
       config =config_alex_pbt if optim=="pbt" else config_alex
    elif arch =='vgg':
       config =config_vgg_pbt if optim=="pbt" else config_vgg
    elif arch =='def':
       pass
    return config

if __name__ == "__main__":
    main(num_samples=100, num_epochs=35, folder="Sort-Dataset", arch='res', optim="asha")
    #main(num_samples=100, num_epochs=35, folder="Sort-Dataset", arch='alex', optim='asha')
    #main(num_samples=40, num_epochs=35, folder="Dataset", arch='alex', opt='pbt')
    #main(num_samples=40, num_epochs=35, folder="Dataset", arch='vgg', opt='pbt')
