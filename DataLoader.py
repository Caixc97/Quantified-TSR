from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
from torch import mean as torch_mean
from tqdm import tqdm
from ConvertImg import GetCoordinate, GetName
import yaml
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

train_data_path = config['train_dataset_path']
test_data_path = config['test_dataset_path']


def get_img_list(path, save_shortcut=False):
    flag_shortcut_exist = os.path.exists(os.path.join(path,'load_shortcut.txt'))
    if flag_shortcut_exist:
        with open(os.path.join(path, 'load_shortcut.txt'), 'r') as file_info:
            result = file_info.readlines()
        result = [item.strip() for item in result]
        success_count = len(result)
    else:
        print('loading...')
        dir_list = [path]
        result = []
        success_count = 0
        while len(dir_list) != 0:
            current_dir = dir_list[0]
            del dir_list[0]
            for file in os.listdir(current_dir):
                file = os.path.join(current_dir, file)
                if os.path.isdir(file):
                    dir_list.append(file)
                elif os.path.isfile(file):
                    if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                        result.append(file)
                        success_count += 1
        if save_shortcut:
            with open(os.path.join(path, 'load_shortcut.txt'), 'w') as file_info:
                for dir_path in result:
                    file_info.write(os.path.abspath(dir_path)+'\n')
    print("%d img loaded successfully" % success_count)
    return result


def default_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class ImageDataset(Dataset):
    COUNT = 0
    CHANGE_COEF = 12800
    POS_INDEX = np.random.randint(0,9)
    while POS_INDEX == 5:
        POS_INDEX = np.random.randint(0, 9)
    def __init__(self, mode='train', img_path=None, transform='default', loader=default_loader, list_segpath=None):
        """
        mode train/test/inference/APP/SSP
        """
        if img_path is not None:
            self.dataset_path = img_path
            self.mode = mode
            print('Data: %s' % img_path)
            self.data_path = get_img_list(self.dataset_path, not mode=='inference')
        else:
            assert(mode == 'train' or mode == 'test' or mode == 'val')
            if mode == 'train':
                print('Data: default train dataset')
                self.mode = 'train'
                self.dataset_path = config['train_dataset_path']
                self.data_path = get_img_list(self.dataset_path, True)
            elif mode == 'test':
                print('Data: default test dataset')
                self.mode = 'test'
                self.dataset_path = config['test_dataset_path']
                self.data_path = get_img_list(self.dataset_path, True)
            elif mode == 'val':
                print('Data: default test dataset')
                self.mode = 'test'
                self.dataset_path = config['val_dataset_path']
                self.data_path = get_img_list(self.dataset_path, True)
        print('Dataloader mode: %s' % mode)
        if transform == 'default':
            normalise = transforms.Normalize(mean=config['data_mean'], std=config['data_std'])
            self.transform = transforms.Compose([transforms.ToTensor(), normalise])
            # self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform
        self.loader = loader
        self.list_segpath = list_segpath

    def __getitem__(self, index):
        self.COUNT += 1
        path = self.data_path[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.mode == 'train' or self.mode == 'test' or self.mode == 'val':
            labels = path[path.rfind(')') + 1:path.rfind('.')]
            target = int(labels)
            target = int(target == 1)
        elif self.mode == 'inference':
            name = os.path.basename(path)
            target = name[:name.rfind('.')]
        elif self.mode == 'RCAM_train':
            if self.COUNT % self.CHANGE_COEF == 0:
                self.change_pos()
            labels = path[path.rfind(')') + 1:path.rfind('.')]
            target = int(labels)
            target = int(target == 1)
            row, col = GetCoordinate(path)
            name = GetName(path)
            img_from_segpath = '.'
            for segpath in self.list_segpath:
                if os.path.exists(os.path.join(segpath, name)):
                    img_from_segpath = os.path.join(segpath, name, os.path.basename(path))
            img_list = [img]
            pos_index = ImageDataset.POS_INDEX
            i = pos_index%3 - 1
            j = pos_index//3 - 1
            path_img = img_from_segpath.replace('('+str(row)+',','('+str(row+i*0.5)+',')\
                .replace(','+str(col)+')',','+str(col+j*0.5)+')')
            path_img = path_img[:path_img.rfind('.jpeg')-1] + '.jpeg'
            if os.path.exists(path_img):
                img_ad = self.loader(path_img)
                if self.transform is not None:
                    img_ad = self.transform(img_ad)
                img_list.append(img_ad)
            else:
                img_ad = np.ones(shape=(256, 256, 3), dtype='uint8') * 255
                if self.transform is not None:
                    img_ad = self.transform(img_ad)
                img_ad[:,max(0,-128*i):256-128*i, max(0,-128*j):256-128*j] = img[:,max(0,128*i):256+128*i, max(0,128*j):256+128*j]
                img_list.append(img_ad)
            return img_list, target, pos_index
        return img, target

    @staticmethod
    def change_pos():
        ImageDataset.POS_INDEX = np.random.randint(0, 9)
        while ImageDataset.POS_INDEX == 5:
            ImageDataset.POS_INDEX = np.random.randint(0, 9)

    def __len__(self):
        return len(self.data_path)

    def count_label(self):
        result = [0 for _ in range(config['num_class'])]
        for path in self.data_path:
            label = int(path[path.rfind(')')+1:path.rfind('.')])
            label = int(label==1)
            result[label] += 1
        return result

    def reserve_hard_data_only(self):
        i = 0
        while i < len(self.data_path):
            path = self.data_path[i]
            label = int(path[path.rfind(')') + 1:path.rfind('.')])
            if label == 0:
                del self.data_path[i]
            else:
                i += 1

def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in tqdm(loader):
        channels_sum += torch_mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch_mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std

def ImageDataLoader(mode,shuffle=False, batch_size=1, num_workers=4,img_path=None,list_segpath=None):
    dataset = ImageDataset(mode=mode, img_path=img_path,list_segpath=list_segpath)
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)

if __name__ == '__main__':
    loader = ImageDataLoader(mode='train',batch_size=100)
    print(loader.dataset.count_label())
    print(get_mean_std(loader))

# file = open('test','wb')
# pickle.dump(img,file)
# file.close()

