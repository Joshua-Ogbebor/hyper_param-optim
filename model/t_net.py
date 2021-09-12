"""Written by Joshua Ogbebor. 
Based on the paper: https://doi.org/10.1111/mice.12263
"""

import torch
import torch.nn as nn
from typing import Union, List, D-ict, Any, cast
from types import SimpleNamespace

#from functools import partial
#from collections import OrderedDict
#from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback


#import torchvision
#from torch.utils.tensorboard import SummaryWriter

__all__ = [
    't_net']
act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "selu": nn.SELU,
    "linear": nn.Identity
}

class t_net(pl.LightningModule):
    '''
    param: channels= 3 for RGB, 1 for grayscale
        '''
    def __init__(
        self,
        config,
        channels:int=3,
        num_classes: int = 4,
        init_weights: bool = True
    ) -> None:                                                           
        super(t_net, self).__init__() 
        self.lr = config["lr"]
        self.momentum = config["mm"]
        self.damp = config["dp"]
        self.wghtDcay = config["wD"]
        self.optim_name=config["opt"]
        self.act_fn_name= config["actvn"] ################################################
        self.act_fn=act_fn_by_name[self.act_fn_name]               
        self.accuracy = torchmetrics.Accuracy()        
        self.losss = nn.CrossEntropyLoss()
        self.betas=(config["b1"],config["b2"])
        self.eps=config["eps"]
        self.rho=config["rho"]
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=24, kernel_size=20, stride=2),                           
            nn.BatchNorm2d(num_features=24))
        self.P1 = nn.MaxPool2d(kernel_size=7, stride=2)
        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=15, stride=2),
            nn.BatchNorm2d(num_features=48))
        self.P2 = nn.MaxPool2d(kernel_size=4, stride=2)
        self.C3 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=10, stride=2),
            nn.BatchNorm2d(num_features=96))                                 # Check if BatchNorm1d              

        self.DropOut = nn.Dropout()                                          # This is just 1d lol
        self.ACT = self.act_fn()                                                          #
        self.C4 = nn.Conv2d(in_channels=96, out_channels=2, kernel_size=1, stride=1)                   # Check if Conv1d
        #self.Linear = nn.Linear(in_features=4, out_features=num_classes)

    def forward(self,x):
        x = self.C1(x)
        x = self.P1(x)
        x = self.C2(x)
        x = self.P2(x)
        x = self.C3(x)
        x = self.DropOut(x)
        x = self.ACT(x)
        x = self.C4(x)
        #x = self.Linear(Output)
        return x


        
        self.features = make_layers(cfg=config['vgg_config'],batch_norm=config['batch_norm'],self.act_fn)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            self.act_fn(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            self.act_fn(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()
        

    def configure_optimizers(self):
        optim={
            'sgd':torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, dampening=self.damp, weight_decay=self.wghtDcay),
            'adam':torch.optim.Adam(self.parameters(), lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.wghtDcay),
            'adadelta':torch.optim.Adadelta(self.parameters(), lr=self.lr, rho=self.rho, eps=self.eps, weight_decay=self.wghtDcay)
        }
        return optim[self.optim_name]

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x.float())
       # print(x,y,logits)
        
        y=y.long()
        loss=self.losss(logits,y)
        
        acc = self.accuracy(logits, y)
        self.log("train_loss", loss,on_step=True, on_epoch=True,sync_dist=True)
        self.log("train_accuracy", acc,on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x.float())
        y=y.long()
        loss=self.losss(logits,y)
        acc = self.accuracy(logits, y)
        self.log("val_loss_init", loss,on_step=True, on_epoch=True,sync_dist=True)
        self.log("val_accuracy_init", acc,on_step=True, on_epoch=True,sync_dist=True)
        return {"val_loss": loss, "val_accuracy": acc}

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(
            [x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack(
            [x["val_accuracy"] for x in outputs]).mean()
        if self.trainer.is_global_zero:
            self.log("val_loss", avg_loss,rank_zero_only=True)
            self.log("val_accuracy", avg_acc,rank_zero_only=True)


 
