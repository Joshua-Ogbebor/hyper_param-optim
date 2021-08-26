import sys
from data import datamodule
from model.residual_net import Resnet_Classifier
from train import fit
import torch
import os
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
import torchmetrics
from ray import tune


    
def main (num_samples=40, num_epochs=50, gpus_per_trial =1, folder="Dataset"):
    os.environ["SLURM_JOB_NAME"] = "bash"
    data_dir = os.path.join(os.getcwd(), folder)

    ########## configs ################
    config_inc = {
        "lr": 1e-4,
        "mm": 0.6,
        "dp":0,
        "wD": 0.000008,
        "depth":tune.choice([1]),
        "actvn":tune.choice(['relu','leaky_relu','selu','linear','tanh']),
        "batch_size": 64
    }
        
        
    config_res = {
        "lr": 1e-4,
        "mm": 0.6,
        "dp":0,
        "wD": 0.000008,
        "bloc_1":tune.choice([64,128,256,512]),
        "bloc_2":tune.choice([64,128,256,512]),
        "bloc_3":tune.choice([64,128,256,512]),
        "bloc_4":tune.choice([64,128,256,512]),
        "depth_1":tune.choice([1,2,0]),
        "depth_2":tune.choice([1,2,0]),
        "depth_3":tune.choice([1,2,0]),
        "depth_4":tune.choice([1,2,0]),
        "actvn":tune.choice(['relu','leaky_relu','selu','linear','tanh']),
        "batch_size": 64,
    }

  
    
    ######### PBT Scheduler ##########################
    

    scheduler = PopulationBasedTraining(
        perturbation_interval=4,
        hyperparam_mutations={
            "lr": tune.loguniform(1e-4, 1e-1),
            "mm":tune.choice([0.6,0.9,1.2]),
            "dp":tune.choice([0,0.9,0.995]),
            "wD":tune.choice([0.000008,0.00001,0.00003 ]),
            "batch_size": [32, 64, 128]
        },
        metric="loss",
        mode="min"
        )
    ##################### Reporter optional ##############
    reporter_inc = CLIReporter(
        parameter_columns=["layer_1_size", "layer_2_size", "lr", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])
    reporter_res = CLIReporter(
        parameter_columns=["layer_1_size", "layer_2_size", "lr", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])


    ############### tune.with_parameters inception net ############
    trainable_inc = tune.with_parameters(
        fit.train_fn_inc_pbt,
        data_dir=data_dir,
        num_epochs=num_epochs,
        num_gpus=1)

    analysis = tune.run(
        trainable_inc,
        resources_per_trial={
            "cpu": 2,
            "gpu": gpus_per_trial
        },
        local_dir="./pbt",
        #verbose=2,
        config=config_inc,
        num_samples=num_samples,
        scheduler=scheduler,
        #progress_reporter=reporter,
        name="tune_inc_pbt")

    print(analysis.best_config)
     
    ########## tune.with_parameters resnet ############
    trainable_res = tune.with_parameters(
        fit.train_fn_res_pbt,
        data_dir=data_dir,
        num_epochs=num_epochs,
        num_gpus=1)

    analysis = tune.run(
        trainable_res,
        resources_per_trial={
           "cpu": 2,
            "gpu": gpus_per_trial 
        },
        local_dir="./pbt",
        #verbose=2,
        config=config_inc,
        num_samples=24,       
        scheduler=scheduler,
        #progress_reporter=reporter,
        name="tune_res_pbt")

    print(f'BEST CONFIG is: { analysis.best_config}')

# Select number of samples, epochs, and dataset folder
if __name__ == "__main__":
    main(num_samples=10, num_epochs=30,gpus_per_trial =1, folder="Dataset")

