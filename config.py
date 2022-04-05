######## ASHA Scheduler #################
scheduler_a = ASHAScheduler(
        max_t=num_epochs,
        metric="loss",
        mode="min",
        grace_period=1,
        reduction_factor=2)

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
        "blk":tune.choice(['Bottleneck','BasicBlock']),
        "groups":tune.choice([32,64,128,256]),
        "wpg":tune.choice([1,2,3,4,5,6,7,8]),
        #"bloc_4":tune.choice([64,128,256,512]),
        #"bloc_2":0,
        #"bloc_3":0,
        #"bloc_4":0,
        "depth_1":tune.choice([1,2,3]),
        "depth_2":tune.choice([1,2,3]),
        "depth_3":tune.choice([1,2,3]),
        "depth_4":tune.choice([1,2,3]),
        #"depth_2":0,
        #"depth_3":0,
        #"depth_4":0,
        "actvn":tune.choice(['relu','leaky_relu','selu','linear','tanh']),
        "batch_size":tune.choice([96, 64, 128]),
        "opt":tune.choice(['adam','sgd', 'adadelta']),
        "b1":tune.choice([0.9]),
        "b2":tune.choice([0.999]),
        "eps":tune.loguniform(1e-08,1e-04),
        "rho":tune.choice([0.9])
    }
