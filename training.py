import glob
import torch
import numpy as np
from torch import nn
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip
from torch.utils.data import Dataset, DataLoader


ROOT_DIR = Path(__file__).parent
DATA_SET = ROOT_DIR / 'dataset' / 'images'
TRAIN_IMAGES = ROOT_DIR / 'dataset' / 'images' / 'train'
VALIDATION_IMAGES = ROOT_DIR / 'dataset' / 'images' / 'validation'

output_type = {}
count_type = [0 for i in range(7)]

for index, val in enumerate(glob.glob(str(TRAIN_IMAGES / '*'))):
    output_type[Path(val).parts[-1]] = index


class Data(Dataset):
    def __init__(self, is_train=True):
        self.is_train = is_train
        data_type = 'train' if is_train else 'validation'
        
        self.path_file = glob.glob(str(DATA_SET / data_type / '*' / '*'))

        if is_train:
            temp_data = []
            for i in range(9):
                self.path_file += glob.glob(str(DATA_SET / data_type / 'disgust' / '*'))
            
            for type in output_type:
                counter = 0
                for path in self.path_file:
                    path_parts = Path(path).parts
                    if len(path_parts) > 6 and path_parts[-2] == type and counter < 7000:
                        counter += 1
                        temp_data.append(path)
                        count_type[output_type[type]] += 1
            self.path_file = temp_data

        self.transform = ToTensor()
        self.data = nn.Sequential(
                                RandomResizedCrop((48, 48),
                                    scale=(0.8, 1),
                                    ratio=(0.5, 1)),
                                RandomHorizontalFlip(),
                                RandomVerticalFlip(),
        )

    def __len__(self):
        return len(self.path_file)

    def __getitem__(self, idx):
        image_path = self.path_file[idx]
        image = Image.open(image_path)
        label = output_type[Path(image_path).parts[-2]]
        image = self.transform(image)
        if self.is_train:
            image = self.data(image)
        
        return (image, label)


training = Data(is_train=True)

for key, val in output_type.items():
    print(f'{key}: {count_type[val]}')

validation_data = Data(is_train=False)
train_dataloader = DataLoader(training, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=64, shuffle=True)
