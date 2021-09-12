import os
import random
import shutil
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader


class ImgData(pl.LightningDataModule):
    def __init__(self, num_workers, batch_size, data_dir: str = os.path.join(os.getcwd(), "Dataset")):
        super().__init__()
        self.data_dir = data_dir
        #_, self.classes = self.dataset( mode='fit')
        #self.mnist_train, self.mnist_val = random_split(mnist_full, [int(0.9*len(data_full)), int(0.1*len(data_full))])
        #self.num_classes=len(self.classes)
        self.num_workers=num_workers
        #self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.batch_size=batch_size
        self.transform = {
            'train': transforms.Compose
            ([
                transforms.Resize(size=256),
                #transforms.RandomRotation(degrees=10),
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomVerticalFlip(),
                #transforms.ColorJitter(brightness=0.1, contrast=0.1),
                #transforms.Normalize(self.mean_nums, self.std_nums),   USE IF COMMAND HERE FOR NORMALIZATION
                transforms.ToTensor()     
            ]), 
            'val': transforms.Compose
            ([
                transforms.Resize(256),
                #transforms.ColorJitter(brightness=0.1, contrast=0.1),
                #transforms.Normalize(self.mean_nums, self.std_nums),                
                transforms.ToTensor()
            ]),
        }
        
        _, self.classes = self.dataset( mode='train')
        #self.mnist_train, self.mnist_val = random_split(mnist_full, [int(0.9*len(data_full)), int(0.1*len(data_full))])
        self.num_classes=len(self.classes)
        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (3, 256, 256)

    def prepare_data(self):
        pass
        # download
        #self.num_classes=self.create_training_data()

    def setup(self, stage:str = 'fit'):
        if stage == 'fit' or stage is None:
            data_full, _ = self.dataset(mode = 'train')
            self.data_train, self.data_val = random_split(data_full, [int(round(len(data_full)*0.9)), int(round(len(data_full)*0.1))])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.data_test,_ = self.dataset( mode='val')

            # Optionally...
            # self.dims = tuple(self.mnist_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size= self.batch_size, num_workers=self.num_workers)# , pin_memory=True)
    def test_dataloater(self):
        return DataLoader(self.data_test,batch_size= self.batch_size, num_workers=self.num_workers )#, pin_memory=True )

    def val_dataloader(self):
        return DataLoader(self.data_val,batch_size= self.batch_size, num_workers=self.num_workers)# , pin_memory=True)
    
    
    def dataset(self, mode):
        data_path = os.path.join(self.data_dir, mode)
        dataset = datasets.ImageFolder(
            root=data_path,
            transform= self.transform[mode])#, target_transform=self.target_to_oh)
        return  dataset , dataset.classes
    
    
    def target_to_oh(self, target):
        one_hot = torch.eye(self.num_classes)[target]
        return one_hot
    
