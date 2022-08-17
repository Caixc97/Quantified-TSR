from model import getModel
from DataLoader import ImageDataLoader
import numpy as np
import torch
import os
from train import get_saved_model
from tqdm import tqdm
import yaml
import sys
from ConvertImg import GetCoordinate

with open('config.yaml', 'r',encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

save_dir = config['train_save_path']
num_class = config['num_class']

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class feature_encoder():
    def __init__(self, model=None, epoch=None, it=None):
        self.flag_cuda = torch.cuda.is_available()
        if model == None:
            self.model = getModel(backbone=config['backbone'], num_class=num_class)
            self.model.load_state_dict(get_saved_model(epoch, it))
        else:
            self.model = model
        self.model.eval()
        print('model state loaded successfully')
        if self.flag_cuda:
            print('gpu mode:gpu')
            self.model = self.model.cuda()
        else:
            print('gpu mode:cpu')

    def get_features(self, data_path, output_path):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        if data_path.endswith('mrxs'):
            slide_name = os.path.basename(data_path[:data_path.rfind('.mrxs')])
            if not os.path.exists(os.path.join(output_path, slide_name)):
                os.mkdir(os.path.join(output_path, slide_name))
            output_path = os.path.join(output_path, slide_name)
            from SlideSeg import seg_slide
            seg_path = os.path.join(output_path, 'seg_temp')
            seg_slide(data_path, seg_path, num_workers=30, overlap=2)
        else:
            slide_name = os.path.basename(data_path)
            seg_path = data_path
        max_row = 0
        max_col = 0
        for file in os.listdir(seg_path):
            if file.endswith('jpeg'):
                row, col = GetCoordinate(file)
                if row > max_row:
                    max_row = row
                if col > max_col:
                    max_col = col
        max_row = int(max_row+0.75)
        max_col = int(max_col+0.75)
        dataloader = ImageDataLoader(mode='inference', shuffle=False, batch_size=1024, num_workers=15,
                                     img_path=seg_path)
        feature_dim = self.model.feature_dim()[2]
        total_features = np.zeros((max_row*2, max_col*2, feature_dim), dtype='float32')
        count = np.zeros((max_row*2, max_col*2), dtype='uint8')
        with torch.no_grad():
            for j, (input, path) in enumerate(dataloader):
                features = self.model.get_features(input.cuda())
                for i in range(input.size(0)):
                    row, col = GetCoordinate(path[i])
                    f = features[i].detach().cpu().numpy()
                    x = total_features[int(row*2):int(row*2)+2, int(col*2):int(col*2)+2].shape
                    total_features[int(row*2):int(row*2)+2, int(col*2):int(col*2)+2] += f[:x[0], :x[1], :]
                    count[int(row*2):int(row*2)+2, int(col*2):int(col*2)+2] += 1
                str_print = "{0:.1f}%".format(j*100 / len(dataloader))
                sys.stdout.write('\r%s' % str_print)
                sys.stdout.flush()
        count[count == 0] = 1
        for i in range(feature_dim):
            total_features[:, :, i] = total_features[:, :, i] / count
        avg_features = np.zeros((max_row, max_col, feature_dim), dtype='float32')
        for row in range(max_row):
            for col in range(max_col):
                avg_features[row, col] = np.mean(
                    total_features[int(row * 2):int(row * 2) + 2, int(col * 2):int(col * 2) + 2], axis=(0, 1))
        blank_img = np.ones((256, 256, 3), dtype='uint8') * 255
        blank_img = dataloader.dataset.transform(blank_img)
        blank_img = torch.unsqueeze(blank_img, axis=0)
        blank_feature = self.model.get_features(blank_img.cuda())[0].detach().cpu().numpy()
        blank_feature = np.mean(blank_feature, axis=(0, 1))
        avg_features[np.max(np.abs(avg_features), axis=2) == 0] = blank_feature
        roi_row = [726, 0]
        roi_col = [325, 0]
        for file in os.listdir(seg_path):
            row, col = GetCoordinate(file)
            if row > roi_row[1]:
                roi_row[1] = row
            if row < roi_row[0]:
                roi_row[0] = row
            if col > roi_col[1]:
                roi_col[1] = col
            if col < roi_col[0]:
                roi_col[0] = col
        roi_row[0] = int(roi_row[0])
        roi_row[1] = int(roi_row[1]+0.5)
        roi_col[0] = int(roi_col[0])
        roi_col[1] = int(roi_col[1]+0.5)
        avg_features[:roi_row[0], :, :] = 0
        avg_features[roi_row[1]:, :, :] = 0
        avg_features[:, :roi_col[0], :] = 0
        avg_features[:, roi_col[1]:, :] = 0
        np.save(os.path.join(output_path, slide_name), avg_features)







if __name__ == '__main__':
    dir_path = 'data/original_slide/seg'
    label_path = 'data/fcn_data/label'
    train_set = set()
    test_set = set()
    val_set = set()
    for file in os.listdir(os.path.join(label_path,'train')):
        if not file.endswith('npy'):
            continue
        train_set.add(file[:file.rfind('.')])
    for file in os.listdir(os.path.join(label_path,'test')):
        if not file.endswith('npy'):
            continue
        test_set.add(file[:file.rfind('.')])
    for file in os.listdir(os.path.join(label_path,'val')):
        if not file.endswith('npy'):
            continue
        val_set.add(file[:file.rfind('.')])
    model = getModel('resnet18')
    model.load_state_dict(get_saved_model())
    classifier = feature_encoder(model=model)
    for file in tqdm(os.listdir(dir_path)):
        if not os.path.isdir(os.path.join(dir_path, file)):
            continue
        if file not in train_set and file not in test_set and file not in val_set:
            continue
        classifier.get_features(os.path.join(dir_path, file), 'data/fcn_data/features')


