from model import getModel
from DataLoader import ImageDataLoader
from evaluate import test_model
from sklearn.metrics import roc_auc_score
import random
from tools import *
import yaml
with open('config.yaml', 'r',encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

save_dir = config['train_save_path'] + '_RCAM'
state_path = os.path.join(save_dir,'state_dict')
b = 0

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)


train_loader = ImageDataLoader(mode='RCAM_train', shuffle=True, batch_size=128, num_workers=16,
                               img_path='data/train/',
                               list_segpath=['data/original_slide/seg', 'data/original_slide/seg_series'])
val_loader = ImageDataLoader(mode='val', shuffle=False, batch_size=1024, num_workers=5,img_path='data/test')
count_label = train_loader.dataset.count_label()
x, y = count_label
weights = torch.tensor([y/(x+y), x/(x+y)]).cuda()
criterion = RCAMLoss(weight=weights)
model = getModel(backbone='resnet34', num_class=config['num_class'])
model = model.cuda()
model.load_state_dict(get_saved_model(0,16787,save_dir='train_result_RCAM'))
model.eval()



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


from sklearn import metrics
import matplotlib.pyplot as plt
fpr, tpr, _ = metrics.roc_curve(y_true,  y_prob)
auc = metrics.roc_auc_score(y_true, y_prob)

plt.plot(fpr, tpr,  color='blue', lw = 2)
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Patch-level receiver operating curve')
plt.legend(['AUC %0.5f' % auc_score], loc="lower right")
plt.savefig('roc.pdf')
