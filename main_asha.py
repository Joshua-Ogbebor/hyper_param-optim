import sys
from data import datamodule
from model.residual_net import Resnet_Classifier
from train import fit
import torch
import os

from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
import torchmetrics

#from ray.tune.integration.pytorch_lightning import TuneReportCallback
#import tempfile
from ray import tune


def main (num_samples=40, num_epochs=50, gpus_per_trial =1, folder="Dataset"):
    os.environ["SLURM_JOB_NAME"] = "bash"
    data_dir = os.path.join(os.getcwd(), folder)
        
    ######## Config for resnet and inception  net ############
    config_res = {

     "lr": tune.loguniform(1e-4, 1e-1),
        "mm":tune.choice([0.6,0.9,1.2]),
        "dp":tune.choice([0,0.9,0.995]),
        "wD":tune.choice([0.000008,0.00001,0.00003 ]),
        "bloc_1":tune.choice([64,128,256,512]),
        "bloc_2":tune.choice([64,128,256,512]),
        "bloc_3":tune.choice([64,128,256,512]),
        "bloc_4":tune.choice([64,128,256,512]),
        "depth_1":tune.choice([1,2,0]),
        "depth_2":tune.choice([1,2,0]),
        "depth_3":tune.choice([1,2,0]),
        "depth_4":tune.choice([1,2,0]),
        "actvn":tune.choice(['relu','leaky_relu','selu','linear','tanh']),
        "batch_size":tune.choice([32, 64, 128]),
    }

    config_inc = {

     "lr": tune.loguniform(1e-4, 1e-1),
        "mm":tune.choice([0.6,0.9,1.2]),
        "dp":tune.choice([0,0.9,0.995]),
        "wD":tune.choice([0.000008,0.00001,0.00003 ]),
        "depth":tune.choice([1]),
        "actvn":tune.choice(['relu','leaky_relu','selu','linear','tanh']),
        "batch_size":tune.choice([32, 64]),
    }
    
    ######## ASHA Scheduler #################
    scheduler = ASHAScheduler(
        max_t=num_epochs,
        metric="loss",
        mode="min",
        grace_period=1,
        reduction_factor=2)
    
    ######### Reporters ########
    reporter_res = CLIReporter(
        parameter_columns=["bloc_1", "bloc_2", "lr", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])

    reporter_inc = CLIReporter(
        parameter_columns=["layer_1_size", "layer_2_size", "lr", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])
    
    ######### tune.with_parameters inception net #######
    trainable1 = tune.with_parameters(
        fit.train_fn_inc,
        data_dir=data_dir,
        num_epochs=num_epochs,
        num_gpus=1)

    analysis = tune.run(
        trainable1,
        resources_per_trial={
            #"cpu": 4,
            "gpu": 1
        },
        local_dir="./results",
        #verbose=2,
        config=config_inc,
        scheduler=scheduler,
        #progress_reporter=reporter_inc,
        num_samples=num_samples,
        name="tune_inc")

    print(analysis.best_config)
    
    ####### tune.with_parameters resnet #######
    trainable2 = tune.with_parameters(
        fit.train_fn_res,
        data_dir=data_dir,
        num_epochs=num_epochs,
        num_gpus=1)

    analysis = tune.run(
        trainable2,
        resources_per_trial={
            #"cpu": 4,
            "gpu": 1
        },
        local_dir="./results",
        config=config_res,
        #verbose=2,
        scheduler=scheduler,
        #progress_reporter=reporter_res,
        num_samples=num_samples,
        name="tune_res")

    print(analysis.best_config)


if __name__ == "__main__":
    main(num_samples=40, num_epochs=35, gpus_per_trial=2, folder="Dataset")
