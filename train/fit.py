from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
import torch
import os
import tempfile
from ray import tune
import pytorch_lightning as pl
import sys
sys.path.append("..")
from model import residual_net 
from model import inception_net
from data import datamodule
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import TensorBoardLogger


####### fit inception net using Asha schediler, or **** #######
def train_fn_inc(config, data_dir=os.path.join(os.getcwd(), "Dataset") , num_epochs=60, num_gpus=0,  checkpoint_dir=None):
    dm = datamodule.ImgData(num_workers=0, batch_size=config["batch_size"], data_dir=data_dir)

    model = inception_net.Googlenet_Classifier(config,  dm.num_classes, data_dir)

    metrics = {"loss": "val_loss", "acc": "val_accuracy"}
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        logger = TensorBoardLogger(save_dir=tune.get_trial_dir(), name="my_model"),
        #log_every_n_steps=5000,
        progress_bar_refresh_rate=0,
        accelerator='ddp',
        plugins=DDPPlugin(find_unused_parameters=False),
        callbacks=[TuneReportCallback(metrics, on="validation_end")])
    trainer.fit(model, dm)

    
###### fit resnet 
def train_fn_res(config, data_dir=os.path.join(os.getcwd(), "Dataset") , num_epochs=60, num_gpus=0,  checkpoint_dir=None):

    dm = datamodule.ImgData(num_workers=0, batch_size=config["batch_size"], data_dir=data_dir)

    model = residual_net.Resnet_Classifier(config, dm.num_classes, data_dir)

    metrics = {"loss": "val_loss", "acc": "val_accuracy"}

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        logger = TensorBoardLogger(save_dir=tune.get_trial_dir(), name="my_model"),
        #log_every_n_steps=5000,
        progress_bar_refresh_rate=0,
        accelerator='ddp',
        plugins=DDPPlugin(find_unused_parameters=False),
        callbacks=[TuneReportCallback(metrics, on="validation_end")])
    trainer.fit(model, dm)


###### fit inception net using PBT scheduler ######
def train_fn_inc_pbt(config, data_dir=os.path.join(os.getcwd(), "Dataset") , num_epochs=60, num_gpus=0,  checkpoint_dir=None):
    dm = datamodule.ImgData(num_workers=0, batch_size=config["batch_size"], data_dir=data_dir)
    
    metrics = {"loss": "val_loss", "acc": "val_accuracy"}
    
    kwargs = {
        "max_epochs": num_epochs,
        # If fractional GPUs passed in, convert to int.
        "gpus": num_gpus,
        "logger": TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        "progress_bar_refresh_rate": 0,
        "callbacks": [
            TuneReportCheckpointCallback(
                metrics=metrics,
                filename="checkpoint",
                on="validation_end")
        ]
        #"accelerator":'ddp',
        #"plugins":DDPPlugin(find_unused_parameters=False),
    }

    if checkpoint_dir:
        kwargs["resume_from_checkpoint"] = os.path.join(
            checkpoint_dir, "checkpoint")
        
    model = inception_net.Googlenet_Classifier(config=config, n_classes=dm.num_classes, data_dir=data_dir)
    trainer = pl.Trainer(**kwargs)
    trainer.fit(model,dm)
    
    
    
    
###### fit resnet using PBT scheduler ########    
def train_fn_res_pbt(config, data_dir=os.path.join(os.getcwd(), "Dataset") , num_epochs=60, num_gpus=0,  checkpoint_dir=None):

    dm = datamodule.ImgData(num_workers=0, batch_size=config["batch_size"], data_dir=data_dir)
    metrics = {"loss": "val_loss", "acc": "val_accuracy"}
    
    kwargs = {
        "max_epochs": num_epochs,
        # If fractional GPUs passed in, convert to int.
        "gpus":num_gpus,
        "logger": TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        "progress_bar_refresh_rate": 0,
        "callbacks": [
            TuneReportCheckpointCallback(
                metrics=metrics,
                filename="checkpoint",
                on="validation_end")
        ]
        #"accelerator":'ddp',
        #"plugins":DDPPlugin(find_unused_parameters=False),
    }

    if checkpoint_dir:
        kwargs["resume_from_checkpoint"] = os.path.join(
            checkpoint_dir, "checkpoint")
        
    model = residual_net.Resnet_Classifier(config, dm.num_classes, data_dir)
    trainer = pl.Trainer(**kwargs)
    trainer.fit(model,dm)

