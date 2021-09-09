import os
import random
import shutil
from torchvision import datasets, transforms
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import numpy as np
from numpy import asarray
from PIL import Image

class ImgData(pl.LightningDataModule):
	""" This class handles the Training and Validation dataset for data saved in a folder in the current working directory. The Dataset folder is expected to contain images classified subfolders The number of subfolders indicates the number of classes while the name of the subfolders are the classnames """ 
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
		#self.train_transforms = self.transform['train']
		#self.val_transforms = self.transform['val']
		#self.test_transforms = self.transform['train']

		_, self.classes = self.dataset()
		self.num_classes=len(self.classes)
		# self.dims is returned when you call dm.size(). Setting default dims here because we know them.
		# Could optionally be assigned dynamically in dm.setup()
		self.dims = (3, 256, 256)

	def prepare_data(self):
		pass

	def setup(self, stage:str = 'fit'):
		data_full, _ = self.dataset()
		self.data_train, self.data_val, self.data_test = random_split(data_full, [int(round(len(data_full)*0.7)), int(round(len(data_full)*0.2)),int(round(len(data_full)*0.1))])

	def train_dataloader(self):
		return DataLoader(self.data_train, batch_size= self.batch_size, num_workers=self.num_workers , pin_memory=True)

	def test_dataloater(self):
		return DataLoader(self.data_test,batch_size= self.batch_size, num_workers=self.num_workers , pin_memory=True )

	def val_dataloader(self):
		return DataLoader(self.data_val,batch_size= self.batch_size, num_workers=self.num_workers , pin_memory=True)

	def target_to_oh(self, target):
		one_hot = torch.eye(self.num_classes)[target]
		return one_hot

	def dataset(self):
		dataset = ImageFolderNp(root=self.data_dir)
		#transform= self.transform['train'], target_transform=self.target_to_oh)
		return  dataset , dataset.classes

class ImageFolderNp(datasets.ImageFolder):
	"""Custom dataset that uses Numpy instead. useful for large datasets.
	Extends torchvision.datasets.ImageFolder
	"""
	def __init__(self,
			root: str,
			#transform: Optional[Callable] = None,
			#target_transform: Optional[Callable] = None,
			#loader: Callable[[str], Any] = default_loader,
			#is_valid_file: Optional[Callable[[str], bool]] = None,
			):
		super(ImageFolderNp, self).__init__(root)#loader, IMG_EXTENSIONS if is_valid_file is None else None,
							#transform=transform,
							#target_transform=target_transform,
							#is_valid_file=is_valid_file)
		self.imgs=np.array(self.imgs)

		# override the __getitem__ method. this is the method that dataloader calls
#	def __getitem__(self, index):
#		# this is what ImageFolder normally returns 
#		sample, target = super(ImageFolderNp, self).__getitem__(index)
#		sample=asarray([sample])#,dtype=np.int64)
#		target=asarray([target])#,dtype=np.int64)
#		# the image file path
#		#path = self.imgs[index][0]
#		# make a new tuple that includes original and the path
#		#tuple_with_path = (original_tuple + (path,))
#		return sample, target
#	def __init__(self, root, loader, extensions, transform=None, target_transform=None):
#		classes, class_to_idx = find_classes(root)
#		samples = make_dataset(root, class_to_idx, extensions)
#		if len(samples) == 0:
#			raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
#			"Supported extensions are: " + ",".join(extensions)))
#
#		self.root = root
#		self.loader = loader
#		self.extensions = extensions
#
#		self.classes = classes
#		self.class_to_idx = class_to_idx
##		self.samples = samples
#	
#		self.transform = transform
#		self.target_transform = target_transform

	def __getitem__(self, index):
		"""
		Args:
		index (int): Index

		Returns:
		tuple: (sample, target) where target is class_index of the target class.
		"""
		path, target = self.samples[index] #sample = self.loader(path)
		sample=np.array(Image.open(path)).transpose()
		if self.transform is not None:
			sample = self.transform(sample)
		if self.target_transform is not None:
			target = self.target_transform(target)

		return sample, target



#class DataIter(Dataset):
#	def __init__(self):
#		self.data_np = np.array([x for x in range(24000000)])
#		self.data = [x for x in range(24000000)]

#	def __len__(self):
#		return len(self.data)

#	def __getitem__(self, idx):
#		data = self.data[idx]
#		data = np.array([data], dtype=np.int64)
#		return torch.tensor(data)
