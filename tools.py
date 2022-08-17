import os
import torch
import sys
import yaml
from torch import nn
import numpy as np
with open('config.yaml', 'r',encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

num_class = config['num_class']

class RCAMLoss(nn.Module):
    def __init__(self, weight=None, alpha_RCAM=1):
        super().__init__()
        self.CrossEntropies = nn.CrossEntropyLoss(weight=weight)
        self.alpha_RCAM = alpha_RCAM
    def forward(self, output, target, cam1=None, cam2=None,  pos_index=0):
        loss1 = self.CrossEntropies(output, target)
        if cam1 == None:
            loss2 = 0
        elif len(torch.unique(pos_index)) != 1:
            loss2 = 0
            # for k in range(cam1.size(0)):
            #     i = pos_index[k] % 3 - 1
            #     j = torch.div(pos_index[k],3,rounding_mode='floor') - 1
            #     loss2 += torch.square(cam1[k, max(0, -4*i):8-4*i, max(0, -4*j):8-4*j, :] -
            #                          cam2[k, max(0, 4*i):8+4*i, max(0, 4*j):8+4*j, :])
            # loss2 = torch.mean(loss2)/cam1.size(0)
        else:
            i = pos_index[0] % 3 - 1
            j = torch.div(pos_index[0], 3, rounding_mode='floor') - 1
            loss2 = torch.square(cam1[:, max(0, -4 * i):8 - 4 * i, max(0, -4 * j):8 - 4 * j, :] -
                                                            cam2[:, max(0, 4*i):8+4*i, max(0, 4*j):8+4*j, :])
            loss2 = loss2.mean()
        loss = loss1 + self.alpha_RCAM * loss2
        return loss

class AverageMeter(object):
    def __init__(self,is_list=False):
        self.is_list=is_list
        self.reset()
    def reset(self):
        if self.is_list:
            self.val = [0 for _ in range(num_class)]
            self.avg = [0 for _ in range(num_class)]
            self.sum = [0 for _ in range(num_class)]
            self.count = 0
        else:
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0
    def update(self, val, n=1):
        if self.is_list:
            self.count += n
            for i in range(num_class):
                self.val[i] = val[i]
                self.sum[i] += val[i] * n
                self.avg[i] = self.sum[i] / self.count
        else:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

def eval(output, target):
    recalls = []
    pres = []
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    for c in range(config['num_class']):
        tp = (pred.eq(c) * target.eq(c)).sum()
        size_pred_true = pred.eq(c).sum()
        size_ground_true = target.eq(c).sum()
        if size_ground_true == 0:
            recall = torch.ones((1,)).cuda()
        else:
            recall = tp/size_ground_true
        if size_pred_true == 0:
            pre = torch.ones((1,)).cuda()
        else:
            pre = tp/size_pred_true
        recalls.append(recall)
        pres.append(pre)
    acc = ((~pred.eq(1) * ~target.eq(1)).sum() + (pred.eq(1) * target.eq(1)).sum())/pred.size(1)
    return recalls, pres, acc


class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = logfile
    def write(self, message):
        self.terminal.write(message+'\n')
        with open(self.log, 'a+') as f:
            f.write(message+'\n')
    def flush(self):
        pass


def get_saved_model(epoch=None,i=None,save_dir=None):
    if save_dir == None:
        save_dir = config['train_save_path']
    state_path = os.path.join(save_dir, 'state_dict')
    if epoch is None:
        epoch = 0
        i = 0
        for file in os.listdir(state_path):
            _epoch = int(file[file.rfind('epoch')+5:file.rfind('_')])
            _i = int(file[file.rfind('_') + 1:file.rfind('.')])
            if _epoch > epoch:
                epoch = _epoch
                i = _i
            if _epoch == epoch and _i > i:
                i = _i
    state_name = 'model_epoch%d_%d.state'%(epoch,i)
    print('model_epoch%d_%d.state'%(epoch,i))
    if torch.cuda.is_available():
        return torch.load(os.path.join(save_dir,'state_dict', state_name))
    else:
        return torch.load(os.path.join(save_dir, 'state_dict', state_name), map_location=torch.device('cpu'))



