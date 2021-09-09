import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict
from torch.nn import functional as F
import pytorch_lightning as pl
import os
import torchmetrics
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback



class Resnet_Classifier(pl.LightningModule):
    def __init__(self, config, n_classes=4, data_dir=None, in_channels = 3):
        super(Resnet_Classifier, self).__init__()
        
        self.data_dir = data_dir or os.path.join(os.getcwd(), "Dataset") 
        self.lr = config["lr"]
        self.momentum = config["mm"]
        self.damp = config["dp"]
        self.wghtDcay = config["wD"]
        self.actvn = config["actvn"]
        self.block_sizes=[config["bloc_1"],config["bloc_2"],config["bloc_3"],config["bloc_4"]]
        self.depths=[config["depth_1"],config["depth_2"],config["depth_3"],config["depth_4"]]
        self.encoder = ResNetEncoder(in_channels,  blocks_sizes=self.block_sizes, depths=self.depths, activation=self.actvn )
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)
        self.betas=(config["b1"],config["b2"])
        self.eps=config["eps"]
        self.rho=config["rho"]     
        self.accuracy = torchmetrics.Accuracy()        
        self.losss = nn.CrossEntropyLoss()
        self.optim_name= config["opt"]
 
    def configure_optimizers(self):
        optim={
            'sgd':torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, dampening=self.damp, weight_decay=self.wghtDcay),
            'adam':torch.optim.Adam(self.parameters(), lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.wghtDcay),
            'adadelta':torch.optim.Adadelta(self.parameters(), lr=self.lr, rho=self.rho, eps=self.eps, weight_decay=self.wghtDcay)
        }
        return optim[self.optim_name]

    def forward(self, x):
        #batch_size, channels, width, height = x.size()
        x = self.encoder(x)
        x = self.decoder(x)
        
        #x = torch.log_softmax(x, dim=1)
        return x

    #def configure_optimizers(self):
        #return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, dampening=self.damp, weight_decay=self.wghtDcay)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x.float())
        #loss = F.nll_loss(logits, y)
        
        y=y.long()
        loss=self.losss(logits,y)
        
        acc = self.accuracy(logits, y)
        self.log("train_loss", loss,sync_dist=True)
        self.log("train_accuracy", acc, sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x.float())
        #loss = F.nll_loss(logits, y)
        y=y.long()
        loss=self.losss(logits,y)
        
        acc = self.accuracy(logits, y)
        self.log("val_loss_init", loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val_accuracy_init", acc, on_step=True, on_epoch=True, sync_dist=True)
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

         







class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size
        
conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)      
        
def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['linear', nn.Identity()],
        ['gelu', nn.GELU()],
        ['tanh', nn.Tanh()]
    ])[activation]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None
        
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

    
def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels))


class ResNetBasicBlock(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )
    


class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
           conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
             activation_func(self.activation),
             conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
             activation_func(self.activation),
             conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )
    

class ResNetLayer(nn.Module):
    """
    A ResNet layer composed by `n` blocks stacked one after the other
    """
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, 
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x
    

class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by layers with increasing features.
    """
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], depths=[2,2,2,2], 
                 activation='relu', block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes
        
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation_func(activation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=depths[0], activation=activation, 
                        block=block,*args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:])]       
        ])
        
        
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x
    

class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x

class ResNet(nn.Module):
    
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def resnet18(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, depths=[2, 2, 2, 2])

def resnet34(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, depths=[3, 4, 6, 3])

def resnet50(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, depths=[3, 4, 6, 3])

def resnet101(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, depths=[3, 4, 23, 3])

def resnet152(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, depths=[3, 8, 36, 3])


def resnetme(in_channels, n_classes, blocks_sizes, deepths):
    return ResNet(in_channels, n_classes,  blocks_sizes, depths, block=ResNetBasicBlock)




