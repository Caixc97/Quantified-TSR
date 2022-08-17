import torch.nn as nn
from torchvision.transforms import Normalize
import torchvision.models as models
import torch.nn.functional as F
import yaml
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


class FineTuneModel(nn.Module):
    def __init__(self, original_model, arch, num_classes, dropout=True):
        super(FineTuneModel, self).__init__()
        self.num_classes = num_classes
        if arch.startswith('resnet'):
            self.features = nn.Sequential(*list(original_model.children())[:-2])
            if arch == 'resnet18' or arch == 'resnet34':
                self.classifier = nn.Sequential(nn.Linear(512, num_classes))
            else:
                self.classifier = nn.Sequential(nn.Linear(2048, num_classes))
            if dropout:
                self.dropout = nn.Dropout(0.5)
            else:
                self.dropout = nn.Identity()
            self.modelName = 'resnet'

    def forward(self, x, RCAM=False):
        feature_map = self.features(x)
        # print f.size()
        if self.training == False:
            x = F.avg_pool2d(feature_map, kernel_size=8, stride=1)
            x = x.view(x.size(0), -1)
        else:
            x = feature_map.view(feature_map.size(0), feature_map.size(1), -1)
            x = self.dropout(x)
            x = x.mean(axis=-1)
        x = self.classifier(x)
        if not RCAM:
            return x
        else:
            return x, feature_map.permute(0, 2, 3, 1)

    def predict(self, x):
        feature_map = self.features(x)
        if self.training == False:
            x = F.avg_pool2d(feature_map, kernel_size=8, stride=1)
            x = x.view(x.size(0), -1)
        else:
            x = feature_map.view(feature_map.size(0), feature_map.size(1), -1)
            x = self.dropout(x)
            x = x.mean(axis=-1)
        y = self.classifier(x)
        f = feature_map
        f = f.transpose(1,2).transpose(2,3)
        f = f.reshape(f.size(0), 64, f.size(-1))
        return y, f

    def get_features(self, x, pool=True):
        x = self.features(x)
        if pool:
            x = F.avg_pool2d(x, kernel_size=4, stride=4)
        x = x.transpose(1, 2).transpose(2, 3)
        return x

    def feature_dim(self):
        return [8, 8, 512]


def getModel(backbone='resnet50', num_class=config['num_class'], dropout=True):
    if backbone == 'resnet50':
        origin_model = models.resnet50(pretrained=True)
    elif backbone == 'resnet18':
        origin_model = models.resnet18(pretrained=True)
    elif backbone == 'resnet152':
        origin_model = models.resnet152(pretrained=True)
    elif backbone == 'resnet34':
        origin_model = models.resnet34(pretrained=True)
    else:
        print("unknown backbone name!")
        return
    return FineTuneModel(origin_model, backbone, num_class, dropout=dropout)

