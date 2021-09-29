"""Edited by Joshua Ogbebor. 
Original was: https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html
"""

import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast
from types import SimpleNamespace

#from functools import partial
#from collections import OrderedDict
#from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

__all__ = [
    'VGG']
act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "selu": nn.SELU,
    "linear": nn.Identity
}

class VGG(pl.LightningModule):

    def __init__(
        self,
        config,
        #features: nn.Module,
        num_classes: int = 4,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        #self.data_dir = data_dir or os.path.join(os.getcwd(), "Dataset") 
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
        self.features = make_layers(cfg=cfgs[config['vgg_config']],act_fn=self.act_fn,batch_norm=config['batch_norm'])
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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]],act_fn, batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), act_fn(inplace=True)]
            else:
                layers += [conv2d, act_fn(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

