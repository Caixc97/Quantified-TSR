from tools import *
import numpy as np
from model import getModel
import tqdm
from DataLoader import ImageDataLoader
from sklearn.metrics import roc_auc_score
import yaml
with open('config.yaml', 'r',encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

test_dir = config['test_dataset_path']

def test_model(model, val_loader, criterion):
    total_losses = AverageMeter()
    length = val_loader.dataset.__len__()
    y_true = np.zeros(shape=(length), dtype='int64')
    y_prob = np.zeros(shape=(length), dtype='float32')
    # pbar = tqdm.pbar(length=length)
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # pbar.update(1)
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)
            total_losses.update(loss.data.item(), input.size(0))
            y_true[val_loader.batch_size * i:val_loader.batch_size * (i + 1)] = target.cpu().numpy()
            y_prob[val_loader.batch_size * i:val_loader.batch_size * (i + 1)] = output[:, 1].detach().cpu().numpy()
    auc_score = roc_auc_score(y_true, y_prob)
    y_pre = (y_prob > 0.5).astype('int64')
    tp = np.sum((y_pre == 1) * (y_true == 1))
    fp = np.sum((y_pre == 1) * (y_true == 0))
    fn = np.sum((y_pre == 0) * (y_true == 1))
    recall = tp/(tp + fn)
    pre = tp/(tp + fp)
    acc = np.sum(y_true==y_pre)/y_true.shape[0]
    return total_losses, recall, pre, acc, auc_score


def test_models(models, loader, criterion, fraction=1):
    total_losses = AverageMeter()
    ending = len(loader)*fraction
    length = loader.dataset.__len__()
    y_true = np.zeros(shape=(length), dtype='int64')
    y_prob = np.zeros(shape=(length), dtype='float32')
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.cuda()
            target = target.cuda()
            output = 0
            for model in models:
                temp = model(input)
                output += temp
            output = output/len(models)
            loss = criterion(output, target)
            total_losses.update(loss.data.item(), input.size(0))
            y_true[loader.batch_size * i:loader.batch_size * (i + 1)] = target.cpu().numpy()
            y_prob[loader.batch_size * i:loader.batch_size * (i + 1)] = output[:, 1].detach().cpu().numpy()
            if i == ending:
                break
    auc_score = roc_auc_score(y_true, y_prob)
    y_pre = (y_prob > 0.5).astype('int64')
    tp = np.sum((y_pre == 1) * (y_true == 1))
    fp = np.sum((y_pre == 1) * (y_true == 0))
    fn = np.sum((y_pre == 0) * (y_true == 1))
    recall = tp/(tp + fn)
    pre = tp/(tp + fp)
    acc = np.sum(y_true==y_pre)/y_true.shape[0]
    return total_losses, recall, pre, acc, auc_score



if __name__ == '__main__':
    model = getModel(backbone='resnet50', num_class=num_class)
    model.load_state_dict(get_saved_model())
    model = model.cuda()
    test_model(model)