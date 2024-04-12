import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import torch
from torch.utils.data import Dataset
import cv2

class CustomDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.classes = os.listdir(root_dir)
        self.classes.sort()
        print("classes name:", self.classes)
        self.images = []
        for clc in self.classes:
            images_name = os.listdir(self.root_dir + "/" + clc)
            self.images += [self.root_dir + "/" + clc + "/" + img_name for img_name in images_name]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            image = self.transforms(image)
        image_class_name = image_path.split("/")[-2]
        label = torch.tensor(self.classes.index(image_class_name))
        return image, label


def load_data(data_path, image_size, batch_size = 16):
    train_dataset_path = data_path + "/train"
    test_dataset_path = data_path + "/test"

    transforms_train = transforms.Compose([transforms.ToPILImage(),
                                           transforms.Resize(image_size),
                                           transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                                           transforms.RandomHorizontalFlip(p=0.5),
                                           transforms.RandomVerticalFlip(p=0.1),
                                           transforms.RandomRotation(degrees=30),
                                           transforms.ToTensor()
                                           ])
    transforms_test = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize(image_size),
                                          transforms.ToTensor()
                                           ])
    train_dataset = CustomDataset(train_dataset_path, transforms= transforms_train)
    test_dataset = CustomDataset(test_dataset_path, transforms=transforms_test)

    print("no of samples in train dataset", len(train_dataset))
    print("no of samples in test dataset", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle= True)
    test_loader = DataLoader(test_dataset, batch_size= batch_size, shuffle= True)
    return train_loader, test_loader
