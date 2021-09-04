"""
Created on Mon May 20 20:32:37 2019
@author: kc
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
import pytorch_lightning as pl
#############################################################################################################
act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU
}
depth_n_features={
                1:64,
                2:64,
                3:96,
                4:96,
                5:96,
                6:96,
                7:128,
                8:128
                
            }
    
###############################################################################3333333333333333333333333333333333333
    
class Google_custom(pl.LightningModule):
    def __init__(self, num_classes=4, depth=1,  act_fn_name="relu", **kwargs):
            super().__init__()
            
            self.hparams.num_classes=num_classes
            self.hparams.act_fn_name=act_fn_name
            self.hparams.act_fn=act_fn_by_name[act_fn_name]
            self.depth=depth
            self._create_network()
            self._init_params()
            
            
    def _create_network(self):
        # A first convolution on the original image to scale up the channel size
        self.input_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            self.hparams.act_fn()
        )
        # Stacking inception blocks
        self.inception_blocks = nn.Sequential(*self.inception_layers())
        # Mapping to classification output
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(depth_n_features[self.depth], self.hparams.num_classes)
        )


    def _init_params(self):
        # Based on our discussion in Tutorial 4, we should initialize the convolutions according to the activation function
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.input_net(x)
        x = self.inception_blocks(x)
        x = self.output_net(x)
        return x
            
        
    def configure_optimizers(self):#####################
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        z = self.input_net(x)
        a=self.inception_blocks(z)
        x_hat = self.output_net(a)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.input_net(x)
        a=self.inception_blocks(z)
        x_hat = self.output_net(a)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)




    def inception_layers(self):
        # Stacking inception blocks
        Blocks=[
            InceptionBlock(64, c_red={"3x3":32,"5x5":16}, c_out={"1x1":16,"3x3":32,"5x5":8,"max":8}, act_fn=self.hparams.act_fn),
            InceptionBlock(64, c_red={"3x3":32,"5x5":16}, c_out={"1x1":24,"3x3":48,"5x5":12,"max":12}, act_fn=self.hparams.act_fn),
            nn.MaxPool2d(3, stride=2, padding=1), # 32x32 => 16x16
            InceptionBlock(96, c_red={"3x3":32,"5x5":16}, c_out={"1x1":24,"3x3":48,"5x5":12,"max":12}, act_fn=self.hparams.act_fn),
            InceptionBlock(96, c_red={"3x3":32,"5x5":16}, c_out={"1x1":16,"3x3":48,"5x5":16,"max":16}, act_fn=self.hparams.act_fn),
            InceptionBlock(96, c_red={"3x3":32,"5x5":16}, c_out={"1x1":16,"3x3":48,"5x5":16,"max":16}, act_fn=self.hparams.act_fn),
            InceptionBlock(96, c_red={"3x3":32,"5x5":16}, c_out={"1x1":32,"3x3":48,"5x5":24,"max":24}, act_fn=self.hparams.act_fn),
            nn.MaxPool2d(3, stride=2, padding=1), # 16x16 => 8x8
            InceptionBlock(128, c_red={"3x3":48,"5x5":16}, c_out={"1x1":32,"3x3":64,"5x5":16,"max":16}, act_fn=self.hparams.act_fn),
            InceptionBlock(128, c_red={"3x3":48,"5x5":16}, c_out={"1x1":32,"3x3":64,"5x5":16,"max":16}, act_fn=self.hparams.act_fn)
        ]
                # Mapping to classification output
        #n=list(range())
        return Blocks[0:self.depth]

        
  #######################################################################################################################3
    
class InceptionBlock(pl.LightningModule):

    def __init__(self, c_in, c_red : dict, c_out : dict, act_fn):
        """
        Inputs:
            c_in - Number of input feature maps from the previous layers
            c_red - Dictionary with keys "3x3" and "5x5" specifying the output of the dimensionality reducing 1x1 convolutions
            c_out - Dictionary with keys "1x1", "3x3", "5x5", and "max"
            act_fn - Activation class constructor (e.g. nn.ReLU)
        """
        super().__init__()

        # 1x1 convolution branch
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(c_in, c_out["1x1"], kernel_size=1),
            nn.BatchNorm2d(c_out["1x1"]),
            act_fn()
        )

        # 3x3 convolution branch
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(c_in, c_red["3x3"], kernel_size=1),
            nn.BatchNorm2d(c_red["3x3"]),
            act_fn(),
            nn.Conv2d(c_red["3x3"], c_out["3x3"], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out["3x3"]),
            act_fn()
        )

        # 5x5 convolution branch
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(c_in, c_red["5x5"], kernel_size=1),
            nn.BatchNorm2d(c_red["5x5"]),
            act_fn(),
            nn.Conv2d(c_red["5x5"], c_out["5x5"], kernel_size=5, padding=2),
            nn.BatchNorm2d(c_out["5x5"]),
            act_fn()
        )

        # Max-pool branch
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(c_in, c_out["max"], kernel_size=1),
            nn.BatchNorm2d(c_out["max"]),
            act_fn()
        )


    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        x_max = self.max_pool(x)
        x_out = torch.cat([x_1x1, x_3x3, x_5x5, x_max], dim=1)
        return x_out
    