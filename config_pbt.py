from ray import tune
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
config_alex_pbt = {
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
config_vgg_pbt = {
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
