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
from model import vgg_net
from model import alex_net
from data import datamodule
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import TensorBoardLogger


def model_name (model_arch):
   return {'inc':inception_net.Googlenet_Classifier,
    'res':residual_net.Resnet_Classifier,
    'alex': alex_net.AlexNet ,
    'vgg': vgg_net.VGG
   }[model_arch]
####### fit net using ASHA scheduler, or random search  #######

def train_fn(config, model_arch, data_dir=os.path.join(os.getcwd(), "Dataset") , num_epochs=60, num_gpus=0, checkpoint_dir=None):
   dm = datamodule.ImgData(num_workers=8, batch_size=config["batch_size"],data_dir=data_dir)
   model = model_name(model_arch)(config,  dm.num_classes)
   metrics = {"loss": "val_loss", "acc": "val_accuracy"}
   pl.seed_everything(42, workers=True)
   trainer = pl.Trainer(
      max_epochs=num_epochs,
      gpus=num_gpus,
      logger = TensorBoardLogger(save_dir=tune.get_trial_dir(), name="my_model"),
      #log_every_n_steps=5000,
      progress_bar_refresh_rate=0,
      accelerator='ddp',
      plugins=DDPPlugin(find_unused_parameters=False),
      deterministic=True,
      callbacks=[TuneReportCallback(metrics, on="validation_end")])

   trainer.fit(model, dm)

###### fit inception net using PBT scheduler ######
def train_fn_pbt(config,model_arch, data_dir=os.path.join(os.getcwd(), "Dataset") , num_epochs=60, num_gpus=0,  checkpoint_dir=None):
   dm = datamodule.ImgData(num_workers=8, batch_size=config["batch_size"], data_dir=data_dir)    
   metrics = {"loss": "val_loss", "acc": "val_accuracy"}
   pl.seed_everything(42, workers=True)
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
        ],
      "accelerator":'ddp',
      "plugins":DDPPlugin(find_unused_parameters=False),
      "deterministic":True
    }
   if checkpoint_dir:
      kwargs["resume_from_checkpoint"] = os.path.join(checkpoint_dir, "checkpoint")
   model = model_name(model_arch)(config,  dm.num_classes, data_dir)   
   trainer = pl.Trainer(**kwargs)
   trainer.fit(model,dm)
    
    
    
    
###### fit resnet using PBT scheduler ########    
