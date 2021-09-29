
"""
Edited by Joshua Ogbebor
Original : https://pytorch.org/vision/stable/_modules/torchvision/models/alexnet.html
"""
import torch
import torch.nn as nn

from typing import Any

#from functools import partial
#from collections import OrderedDict
#from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback


__all__ = ['AlexNet', 'alexnet']

act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "selu": nn.SELU,
    "linear": nn.Identity
}

class AlexNet(pl.LightningModule):

    def __init__(self,
                 config,
                 num_classes: int = 4,
        ) -> None:
        super(AlexNet, self).__init__()
        self.lr = config["lr"]
        self.momentum = config["mm"]
        self.damp = config["dp"]
        self.wghtDcay = config["wD"]
        self.optim_name=config["opt"]
        self.act_fn_name= config["actvn"]
        self.act_fn=act_fn_by_name[self.act_fn_name]               
        self.accuracy = torchmetrics.Accuracy()        
        self.losss = nn.CrossEntropyLoss()
        self.betas=(config["b1"],config["b2"])
        self.eps=config["eps"]
        self.rho=config["rho"]
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            self.act_fn(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            self.act_fn(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            self.act_fn(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            self.act_fn(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            self.act_fn(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            self.act_fn(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            self.act_fn(),
            nn.Linear(4096, num_classes),
        )
        
        

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
       # :wqprint(x,y,logits)
        
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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


