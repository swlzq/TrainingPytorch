# author: LiuZhQ
# time  : 2019/7/1


import os
import csv
import shutil
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

__all__ = ['DRIVER', 'data_loader']


class DRIVER(Dataset):
    def __init__(self, opts, train=True):
        """
        Dataset is divided train set and val set by different drives.
        :param data_path: dataset root path
        :param input_size: image input size
        :param train: if true, load train set, else load validate set
        """
        self.data_path = opts.data_path
        self.input_size = opts.input_size
        self.train = train
        self.drivers = []
        self.images = []
        self.targets = []
        self.transform = None
        self.size = []

        # Read label csv and get all drivers
        csv_path = os.path.join(self.data_path, 'driver_images_list.csv')
        data_frame = pd.read_csv(csv_path)
        drivers_labels = data_frame['subject'].drop_duplicates().values
        drivers_labels = random.sample(list(drivers_labels), len(drivers_labels))
        # train : val = 8 : 2
        len_validate = int(0.2 * len(drivers_labels))
        if self.train:
            labels = drivers_labels[:-len_validate]  # train set
        else:
            labels = drivers_labels[-len_validate:]  # validate set

        for label in labels:
            df = data_frame[(data_frame['subject'] == label)]
            for _, row in df.iterrows():
                self.drivers.append(row['subject'])
                self.images.append(row['img'])
                self.targets.append(row['classname'])

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if self.train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(self.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                normalize
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        driver = self.drivers[index]
        image_name = self.images[index]
        target = self.targets[index]
        image_path = os.path.join(self.data_path, 'images', driver, target, image_name)
        image = Image.open(image_path)
        image = self.transform(image)
        target = int(target[-1:])
        return image, target


# divide different drivers images into different directions
def create_images_dir():
    data_path = 'F:/PCL/IntelligentTransport/state-farm-distracted-driver-detection'
    csv_path = os.path.join(data_path, 'driver_imgs_list.csv')
    data_frame = pd.read_csv(csv_path)
    drivers_labels = data_frame['subject'].drop_duplicates().values
    img_dir = os.path.join(data_path, 'images')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    for label in drivers_labels:
        driver_path = os.path.join(img_dir, label)
        if not os.path.exists(driver_path):
            os.makedirs(driver_path)
        df = data_frame[(data_frame['subject'] == label)]
        for _, row in df.iterrows():
            old_path = os.path.join(data_path, 'imgs', 'train', row['classname'], row['img'])
            new_path = os.path.join(driver_path, row['classname'])
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            shutil.copy(old_path, new_path)


def create_label_file():
    data_path = 'F:/PCL/IntelligentTransport/state-farm-distracted-driver-detection'
    label_file = os.path.join(data_path, 'driver_images_list.csv')
    with open(label_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['subject', 'classname', 'img'])
        data_path = os.path.join(data_path, 'images')
        driver_list = os.listdir(data_path)
        for driver in driver_list:
            driver_path = os.path.join(data_path, driver)
            category_list = os.listdir(driver_path)
            for category in category_list:
                category_path = os.path.join(driver_path, category)
                image_list = os.listdir(category_path)
                for image in image_list:
                    writer.writerow([driver, category, image])


if __name__ == '__main__':
    data_loader('F:/PCL/IntelligentTransport/state-farm-distracted-driver-detection', train=False)
    # create_label_file()
