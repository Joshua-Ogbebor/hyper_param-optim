import sys
from data import datamodule
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

    ######### Reporters ########
    #reporter_res = CLIReporter(
     #   parameter_columns=["bloc_1", "bloc_2", "lr", "batch_size"],
     #   metric_columns=["loss", "mean_accuracy", "training_iteration"])

    #reporter_inc = CLIReporter(
     #   parameter_columns=["layer_1_size", "layer_2_size", "lr", "batch_size"],
     #   metric_columns=["loss", "mean_accuracy", "training_iteration"])

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
            "cpu": 24,
            "gpu": 1
        },
        local_dir="../analyses/results",
        #verbose=2,
        config=config_dict(arch,optim),
        scheduler=scheduler_switch[optim],# if optim,
       # metric="loss" if not optim,
       # mode="min" if not optim,
        #progress_reporter=reporter_inc,
        num_samples=num_samples,
        name="tune-"+arch+"-"+optim)

    print(analysis.best_config)

def config_dict (arch,optim):
     ######## Config for architectures ############
    config_inc = {

        "lr": tune.loguniform(1e-4, 1e-1),
        "mm":tune.choice([0.6,0.9,1.2]),
        "dp":tune.choice([0,0.9,0.995]),
        "wD":tune.choice([0,0.000008,0.00001,0.00003 ]),
        "depth":tune.choice([1,2,3,4,5]),
        "actvn":tune.choice(['relu','leaky_relu','selu','linear','tanh']),
        "batch_size":tune.choice([48,64,96]),
        "opt": tune.choice(['adam','sgd', 'adadelta']),
        "b1": 0.9,
        "b2":0.999 ,
        "eps":tune.loguniform(1e-08 ,1e-04),
        "rho":0.9
    }
    config_vgg = {

        "lr": tune.loguniform(1e-4, 1e-1),
        "mm":tune.choice([0.6,0.9,1.2]),
        "dp":tune.choice([0,0.9,0.995]),
        "wD":tune.choice([0,0.000008,0.00001,0.00003 ]),
        "vgg_config":tune.choice(['A','B','D','E']),
        "actvn":tune.choice(['relu','leaky_relu','selu','linear','tanh']),
        "batch_size":tune.choice([48,64,96]),
        "opt": tune.choice(['adam','sgd', 'adadelta']),
        "b1":0.9,
        "b2":0.999 ,
        "eps":tune.loguniform(1e-08 ,1e-04),
        "batch_norm": tune.choice([True,False]),
        "rho":0.9
    }
    config_alex = {

        "lr": tune.loguniform(1e-4, 1e-1),
        "mm":tune.choice([0.6,0.9,1.2]),
        "dp":tune.choice([0,0.9,0.995]),
        "wD":tune.choice([0,0.000008,0.00001,0.00003 ]),
        #"depth":tune.choice([1,2,3,4,5]),
        "actvn":tune.choice(['relu','leaky_relu','selu','linear','tanh']),
         "batch_size":tune.choice([48,64,96]),
        "opt": tune.choice(['adam','sgd', 'adadelta']),
        "b1": 0.9,
        "b2":0.999 ,
        "eps":tune.loguniform(1e-08 ,1e-04),
        "rho":0.9
    }


    config_res = {

        "lr":tune.loguniform(1e-4, 1e-1),
        "mm":tune.choice([0.6,0.9,1.2]),
        "dp":tune.choice([0,0.9,0.995]),
        "wD":tune.choice([0,0.000008,0.00001,0.00003 ]),
        "bloc_1":tune.choice([64,128,256,512]),
        "bloc_2":tune.choice([64,128,256,512]),
        "bloc_3":tune.choice([64,128,256,512]),
        "bloc_4":tune.choice([64,128,256,512]),
        #"bloc_2":0,
        #"bloc_3":0,
        #"bloc_4":0,
        "depth_1":tune.choice([1,2,3]),
        "depth_2":tune.choice([1,2,3]),
        #"depth_3":tune.choice([1,2,3]),
        #"depth_4":tune.choice([1,2,3]),
        #"depth_2":0,
        "depth_3":0,
        "depth_4":0,
        "actvn":tune.choice(['relu','leaky_relu','selu','linear','tanh']),
        "batch_size":tune.choice([96, 64, 128]),
        "opt":tune.choice(['adam','sgd', 'adadelta']),
        "b1":tune.choice([0.9]),
        "b2":tune.choice([0.999]),
        "eps":tune.loguniform(1e-08,1e-04),
        "rho":tune.choice([0.9])
    }
    config_inc_pbt = {
        "lr": 1e-4,
        "mm": 0.6,
        "dp":0,
        "wD": 0.000008,
        "depth":tune.choice([1,2,3]),
        "actvn":tune.choice(['relu','leaky_relu','selu','linear','tanh']),
        "batch_size": 64,
        "opt": tune.choice(['adam','sgd', 'adadelta']),
        "b1": 0.9,
        "b2": tune.choice([0.999]),
        "eps":tune.loguniform(1e-08 ,1e-04),
        "rho":tune.choice([0.9])

    }
    config_inc_alex = {
        "lr": 1e-4,
        "mm": 0.6,
        "dp":0,
        "wD": 0.000008,
        #"depth":tune.choice([1,2,3]),
        "actvn":tune.choice(['relu','leaky_relu','selu','linear','tanh']),
        "batch_size": 64,
        "opt": tune.choice(['adam','sgd', 'adadelta']),
        "b1": 0.9,
        "b2": 0.999,
        "eps":tune.loguniform(1e-08 ,1e-04),
        "rho":0.9

    }
    config_inc_vgg = {
        "lr": 1e-4,
        "mm": 0.6,
        "dp":0,
        "wD": 0.000008,
        "vgg_config":tune.choice(['A','B','D','E']),
        "actvn":tune.choice(['relu','leaky_relu','selu','linear','tanh']),
        "batch_size": 64,
        "opt": tune.choice(['adam','sgd', 'adadelta']),
        "b1": 0.9,
        "b2": 0.999,
        "eps":tune.loguniform(1e-08 ,1e-04),
        "rho":0.9,
        "batch_norm": tune.choice([True,False]),

    }

    config_res_pbt = {
        "lr": 1e-4,
        "mm": 0.6,
        "dp":0,
        "wD": 0.000008,
        "bloc_1":tune.choice([64,128,256,512]),
        "bloc_2":tune.choice([64,128,256,512]),
        "bloc_3":tune.choice([64,128,256,512]),
        "bloc_4":tune.choice([64,128,256,512]),
        "depth_1":tune.choice([1,2]),
        "depth_2":tune.choice([1,2,0]),
        "depth_3":tune.choice([1,2,0]),
        "depth_4":0,#tune.choice([1,2,0]),
        "actvn":tune.choice(['relu','leaky_relu','selu','linear','tanh']),
        "batch_size": 64,
        "opt": tune.choice(['adam','sgd', 'adadelta']),
        "b1": 0.9,
        "b2": 0.999,
        "eps":tune.loguniform(1e-08 ,1e-04),
        "rho":0.9
    }

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
    main(num_samples=100, num_epochs=35, folder="Dataset_new", arch='alex', optim="asha")
    #main(num_samples=100, num_epochs=35, folder="Dataset_new", arch='alex', optim='asha')
    #main(num_samples=40, num_epochs=35, folder="Dataset", arch='alex', opt='pbt')
    #main(num_samples=40, num_epochs=35, folder="Dataset", arch='vgg', opt='pbt')

    
