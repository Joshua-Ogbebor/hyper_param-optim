import os
import random
import shutil
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader


class ImgData(pl.LightningDataModule):
     """ This class handles the Training and Validation dataset for data saved in a folder in the 
    current working directory. The Dataset folder is expected to contain images classified subfolders
    The number of subfolders indicates the number of classes while the name of the subfolders are the classnames """ 
        
    def __init__(self, num_workers, batch_size, data_dir: str = os.path.join(os.getcwd(), "Dataset")):
        super().__init__()
        self.data_dir = data_dir
        self.num_workers=num_workers
        self.batch_size=batch_size
        self.transform = {
            'train': transforms.Compose
            ([
                transforms.Resize(size=256),
                #transforms.RandomRotation(degrees=10),
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
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
        self.train_transforms = self.transform['train']
        self.val_transforms = self.transform['val']
        self.test_transforms = self.transform['train']
        
        _, self.classes = self.dataset( mode='train')
        self.num_classes=len(self.classes)
        # self.dims is returned when you call dm.size(). Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (3, 256, 256)

    def prepare_data(self):
        list_dirs_full_init= [file.path for file in os.scandir(dataset_dir) if file.is_dir()]
        i=0
        list_dirs_init=[]
        for dirs in list_dirs_full_init:
            list_dirs_init.append(os.path.split(dirs)[1])
            i=1+i
        self.num_classes = self.list_data_info(list_dirs_full_init)
        
    def setup(self, stage:str = 'fit'):
        data_full, _ = self.dataset
        self.data_train, self.data_val, self.data_test = random_split(data_full, [int(round(len(data_full)*0.7)), int(round(len(data_full)*0.2)),int(round(len(data_full)*0.1))])
        
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size= self.batch_size, num_workers=self.num_workers)# , pin_memory=True)
    
    def test_dataloater(self):
        return DataLoader(self.data_test,batch_size= self.batch_size, num_workers=self.num_workers )#, pin_memory=True )
    
    def val_dataloader(self):
        return DataLoader(self.data_val,batch_size= self.batch_size, num_workers=self.num_workers)# , pin_memory=True)
    
    def dataset(self):
        dataset = datasets.ImageFolder(
            root=self.data_dir)
            #transform= self.transform['train'], target_transform=self.target_to_oh)
        return  dataset , dataset.classes
    
    def target_to_oh(self, target):
        one_hot = torch.eye(self.num_classes)[target]
        return one_hot
    
