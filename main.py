import sys
from data import datamodule
from model.residual_net import Resnet_Classifier
from train import fit
import torch
import os
#from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
import torchmetrics
from ray import tune


def main (num_samples=40, num_epochs=50, folder="Dataset", arch='inc',optim=None):
    os.environ["SLURM_JOB_NAME"] = "bash"
    data_dir = os.path.join(os.getcwd(), folder)
       
    ######## Config for architectures ############
    config_inc = {

        "lr": tune.loguniform(1e-4, 1e-1),
        "mm":tune.choice([0.6,0.9,1.2]),
        "dp":tune.choice([0,0.9,0.995]),
        "wD":tune.choice([0.000008,0.00001,0.00003 ]),
        "depth":tune.choice([1,2,3]),
        "actvn":tune.choice(['relu','leaky_relu','selu','linear','tanh']),
        "batch_size":tune.choice([16,32,48]),
        "opt": tune.choince(['adam','sgd', 'adadelta']),
        "b1": ,
        "b2": ,
        "rho": ,
        "eps":
    }
    config_res = {

        "lr": tune.loguniform(1e-4, 1e-1),
        "mm":tune.choice([0.6,0.9,1.2]),
        "dp":tune.choice([0,0.9,0.995]),
        "wD":tune.choice([0.000008,0.00001,0.00003 ]),
        "bloc_1":tune.choice([64,128,256,512]),
        "bloc_2":tune.choice([64,128,256,512]),
        #"bloc_3":tune.choice([64,128,256,512]),
        #"bloc_4":tune.choice([64,128,256,512]),
        #"bloc_2":0,
        "bloc_3":0,
        "bloc_4":0,
        "depth_1":tune.choice([1,2,3]),
        "depth_2":tune.choice([1,2,3]),
        #"depth_3":tune.choice([1,2,3]),
        #"depth_4":tune.choice([1,2,3]),
        #"depth_2":0,
        "depth_3":0,
        "depth_4":0,
        "actvn":tune.choice(['relu','leaky_relu','selu','linear','tanh']),
        "batch_size":tune.choice([32, 64, 128]),
         "opt": ,
        "b1": ,
        "b2": ,
        "rho": ,
        "eps":
    }
    
    
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
            "batch_size": [32, 48]
        },
        metric="loss",
        mode="min"
        )
    ###### scheduler switcher ##########
    scheduler_switch={
        "asha":scheduler_a,
        "pbt":scheduler_pbt
    }
    
    ######### Reporters ########
    #reporter_res = CLIReporter(
     #   parameter_columns=["bloc_1", "bloc_2", "lr", "batch_size"],
     #   metric_columns=["loss", "mean_accuracy", "training_iteration"])

    #reporter_inc = CLIReporter(
     #   parameter_columns=["layer_1_size", "layer_2_size", "lr", "batch_size"],
     #   metric_columns=["loss", "mean_accuracy", "training_iteration"])
    
    ######### tune.with_parameters inception net #######
    trainable1 = tune.with_parameters(
        fit.train_fn,
        data_dir=data_dir,
        num_epochs=num_epochs,
        num_gpus=1)
    analysis = tune.run(
        trainable1,
        resources_per_trial={
            #"cpu": 4,
            "gpu": 1
        },
        local_dir="../analysis/results",
        #verbose=2,
        config=config_inc,
        model_arch=arch,
        scheduler=scheduler_switch[optim] if optim,
        metric="loss" if not optim,
        mode="min" if not optim,
        #progress_reporter=reporter_inc,
        num_samples=num_samples,
        name="tune-"+arch+"-"+optim)

    print(analysis.best_config)


if __name__ == "__main__":
    main(num_samples=40, num_epochs=35, folder="Dataset", arch='res', optim="asha")
    main(num_samples=40, num_epochs=35, folder="Dataset", arch='inc', optim='asha')
    #main(num_samples=40, num_epochs=35, folder="Dataset", arch='alex', opt='asha')
    #main(num_samples=40, num_epochs=35, folder="Dataset", arch='vgg', opt='asha')

