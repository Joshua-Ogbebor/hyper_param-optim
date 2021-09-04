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
    
    
            
    def create_training_data(self):
        """ This function creates the Training and Validation dataset for data saved in a folder in the 
        current working directory. The Dataset folder is expected to contain labelled images in subfolders
        The number of subfolders indicates the number of classes while the name of the subfolders are the classnames """
        
        #Check if already sorted
        list_dirs_init= [f.path for f in os.scandir(self.data_dir) if f.is_dir()]
        i=0
        for dirs in list_dirs_init:
            list_dirs_init[i] = dirs.split('/')[-1]
            i=1+i

        if not list_dirs_init.sort()== ['train','val']:  ## train only nko? ## some other laabel eg train and test
            #print ("..... ")
            list_dirs=[f.path for f in os.scandir(self.data_dir) if f.is_dir()]    
            Num_classes = len(list_dirs)
        else:
            print ("Previously created")
            list_dirs=[f.path for f in os.scandir(f"{self.data_dir}/{list_dirs_init[0]}") if f.is_dir()]
            Num_classes = len(list_dirs)

        for dirs in list_dirs: 
            class_name = dirs.split('/')[-1]
            train_dir = f"{self.data_dir}/train/{class_name}"                    # Create folder names
            val_dir = f"{self.data_dir}/val/{class_name}" 

            if not list_dirs_init.sort() == ['train','val']:  
            ## if element of listdirs_init==train_dir    OOOORRR   list_dirs_init [0]== 'train':,,,,,,else:print(error?)
                # Training set
                shutil.move(dirs, train_dir)
                # Validation set
                os.makedirs(val_dir)    ## if element of listdirs_init==val_dir
                for f in os.listdir(train_dir):
                    if random.random() > 0.80: #generates a random float uniformly in the semi-open range [0.0, 1.0)
                        shutil.move(f'{train_dir}/{f}', val_dir)
            num_T=len(os.listdir(train_dir))
            num_V=len(os.listdir(val_dir)) 

            # Print number of classes                

            print("{:s}: {:.0f} training images and {:.0f} validation images ({:.1f}%) ".format(class_name, num_T, num_V, 100*num_V/(num_T+num_V)))   
        #print("There are {:.0f} Classes.".format(Num_classes))
        return Num_classes
   

    def create_training_data(self):
        """ This function creates the Training and Validation dataset for data saved in a folder in the 
        current working directory. The Dataset folder is expected to contain labelled images in subfolders
        The number of subfolders indicates the number of classes while the name of the subfolders are the classnames """

        #Check if already sorted
        list_dirs_init= [f.path for f in os.scandir(self.data_dir) if f.is_dir()]
        i=0
        for dirs in list_dirs_init:
            list_dirs_init[i] = dirs.split('/')[-1]
            i=1+i

        if not list_dirs_init.sort()== ['train','val']:  ## train only nko? ## some other laabel eg train and test
            #print ("..... ")
            list_dirs=[f.path for f in os.scandir(self.data_dir) if f.is_dir()]
            Num_classes = len(list_dirs)
        else:
            print ("Previously created")
            list_dirs=[f.path for f in os.scandir(f"{self.data_dir}/{list_dirs_init[0]}") if f.is_dir()]
            Num_classes = len(list_dirs)

        for dirs in list_dirs:
            class_name = dirs.split('/')[-1]
            train_dir = f"{self.data_dir}/train/{class_name}"                    # Create folder names
            val_dir = f"{self.data_dir}/val/{class_name}"

            if not list_dirs_init.sort() == ['train','val']:
            ## if element of listdirs_init==train_dir    OOOORRR   list_dirs_init [0]== 'train':,,,,,,else:print(error?)
                # Training set
                shutil.move(dirs, train_dir)
                # Validation set
                os.makedirs(val_dir)    ## if element of listdirs_init==val_dir
                for f in os.listdir(train_dir):
                    if random.random() > 0.80: #generates a random float uniformly in the semi-open range [0.0, 1.0)
                        shutil.move(f'{train_dir}/{f}', val_dir)
            num_T=len(os.listdir(train_dir))
            num_V=len(os.listdir(val_dir))

            # Print number of classes                

            print("{:s}: {:.0f} training images and {:.0f} validation images ({:.1f}%) ".format(class_name, num_T, num_V, 100*num_V/(num_T+num_V)))
        #print("There are {:.0f} Classes.".format(Num_classes))
        return Num_classes

    
def sort_learning_data():
    """ This function creates the Training and Validation dataset for data saved in a folder in the 
    current working directory. The Dataset folder is expected to contain images classified subfolders
    The number of subfolders indicates the number of classes while the name of the subfolders are the classnames """    
    
    #Check if already labelled 'train' and 'val'
    list_dirs_full_init= [file.path for file in os.scandir(dataset_dir) if file.is_dir()]
    i=0
    list_dirs_init=[]
    for dirs in list_dirs_full_init:
        list_dirs_init.append(os.path.split(dirs)[1])
        i=1+i
    # first if  
    if not all(x in list_dirs_init for x in ['train','val']):    
        #check if it may be labelled already for traininn
        #if there are subdirectories that are same, how can we know which is train or val or test? the user wil be prompted to make this change
        sub_dir=[file.path for file in os.scandir(list_dirs_full_init[0]) if file.is_dir()]
        if len(sub_dir) > 1 :
            if [file.path for file in os.scandir(list_dirs_full_init[0]) if file.is_dir()].sort()==[file.path for file in os.scandir(list_dirs_full_init[1]) if file.is_dir()]:
                print ("Previously labelled with some other labels. \n Please label as 'train' and 'val' accordingly. May cause errors in main program")
            #one or more may be train and will raise errors
            return 
        elif not (sub_dir):
            print("Sorting into train and validation sets")
 #           try:
            os.mkdir(os.path.join(dataset_dir,"train"))
   #             print("folder 'val' created ")
    #        except FileExistsError:
     #           print("folder train already exists")
      #      try:
            os.mkdir(os.path.join(dataset_dir,"val"))
        #        print("folder 'val' created ")
         #   except FileExistsError:
          #      print("folder val already exists")
                            
            for class_path in list_dirs_full_init : 
                _,class_name = os.path.split(class_path)
            
                train_dir = os.path.join(dataset_dir,"train",class_name)                    
                val_dir = os.path.join(dataset_dir,"val",class_name)
                    
                shutil.move(class_path, train_dir)
                
                    
                for file in os.listdir(train_dir):
                    if random.random() > 0.80: #generates a random float uniformly in the semi-open range [0.0, 1.0)
                        shutil.move(os.path.join(train_dir,file), val_dir)
        else:
            print("re-check the dataset folder")
            return
         
    else:
        print ("Previously sorted")
        
    learning_set= [file.path for file in os.scandir(dataset_dir) if file.is_dir()]    
    class_paths=[file.path for file in os.scandir(learning_set[0]) if file.is_dir()] 
    list_data_info(dataset_dir,class_paths)
    
def list_data_info(dataset_dir,class_paths):
    Num_classes = len(class_paths)
    for dirs in class_paths: 
        _,class_name = os.path.split(dirs)            
        train_dir = os.path.join(dataset_dir,"train",class_name)                    
        val_dir = os.path.join(dataset_dir,"val",class_name)
           
        # Print number of classes
        num_T=len(os.listdir(train_dir))
        num_V=len(os.listdir(val_dir))         
        print("{:s}: {:.0f} training images and {:.0f} validation images ({:.1f}%) ".format(class_name, num_T, num_V, 100*num_V/(num_T+num_V)))   
    print("There are {:.0f} Classes.".format(Num_classes))
    return Num_classes

