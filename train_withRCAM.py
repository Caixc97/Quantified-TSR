from model import getModel
from DataLoader import ImageDataLoader
from evaluate import test_model
from torch import optim
import random
import time
from tools import *
import yaml
with open('config.yaml', 'r',encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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

def _train_check(model,total_losses,epoch,i,length,avg_recall,avg_pre,avg_acc, val_loader, criterion):
    if not os.path.exists(os.path.join(save_dir, 'state_dict')):
        os.mkdir(os.path.join(save_dir, 'state_dict'))
    torch.save(model.state_dict(), os.path.join(save_dir, 'state_dict', 'model_epoch%d_%d.state' % (epoch, i)))
    # train msg
    logger.write('epoch {0}:{1}/{2}'.format(epoch,i,length))
    logger.write('\tTrain:\n'
                 '\t\tLoss {loss.avg:.4f}'
                 .format(loss=total_losses))
    msg = '\t\tRecall/Prec:'
    for i in range(config['num_class']):
        msg += str(np.around(avg_recall.avg[i], 3)) + '/' + str(np.around(avg_pre.avg[i], 3)) + ', '
    logger.write(msg)
    logger.write('\t\tAcc:{acc.avg:.4f}'.format(acc=avg_acc))
    # test msg
    model.eval()
    test_loss, test_recall, test_pre, test_acc, test_auc = test_model(model, val_loader, criterion)
    model.train()
    logger.write('\tTest:\n'
                 '\t\tLoss {loss.avg:.4f}'.format(loss=test_loss))
    msg = '\t\tRecall/Prec:'
    for i in range(config['num_class']):
        msg += str(np.around(test_recall, 3)) + '/' + str(np.around(test_pre, 3)) + ', '
    logger.write(msg)
    logger.write('\t\tAcc:{acc:.4f}'.format(acc=test_acc))
    logger.write('\t\tAuc:{auc:.4f}'.format(auc=test_auc))
    total_losses.reset()
    avg_pre.reset()
    avg_recall.reset()
    avg_acc.reset()



def train(train_loader, val_loader, model, criterion, optimizer, epoch, logger):
    avg_recall = AverageMeter(is_list=True)
    avg_pre = AverageMeter(is_list=True)
    avg_acc = AverageMeter()
    total_losses = AverageMeter()
    logger.write('epoch %d' % (epoch))
    model.train()
    for i, (input_list, target, pos_index) in enumerate(train_loader):
        input_var1 = torch.autograd.Variable(input_list[0].cuda())
        input_var2 = torch.autograd.Variable(input_list[1].cuda())
        target = target.cuda()
        output, cam1 = model(input_var1, True)
        _, cam2 = model(input_var2, True)
        target_var = torch.autograd.Variable(target)
        loss = criterion(output, target_var, cam1, cam2, pos_index)
        total_losses.update(loss.data.item(), input_list[0].size(0))
        recall, pre, acc = eval(output, target)
        avg_recall.update([r.data.item() for r in recall], input_list[0].size(0))
        avg_pre.update([p.data.item() for p in pre], input_list[0].size(0))
        avg_acc.update(acc.data.item(), input_list[0].size(0))
        loss = (loss - b).abs() + b
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if i%(len(train_loader)//10) == 0:
        #     print(time.strftime('%H:%M:%S: \t', time.localtime()) +
        #           "Batch {i} loss: {l:.4f}, acc: {a:.4f}".format(i=i, l=total_losses.avg, a=avg_acc.avg))
    _train_check(model, total_losses, epoch, i, len(train_loader), avg_recall, avg_pre, avg_acc, val_loader, criterion)


if __name__ == '__main__':
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    logfile = os.path.join(save_dir, 'log')
    logger = Logger(logfile)
    train_loader = ImageDataLoader(mode='RCAM_train', shuffle=True, batch_size=64, num_workers=16,
                                   img_path='data/train/',
                                   list_segpath=['data/original_slide/seg', 'data/original_slide/seg_series'])
    val_loader = ImageDataLoader(mode='val', shuffle=False, batch_size=1024, num_workers=5,img_path='data/val')
    count_label = train_loader.dataset.count_label()
    x, y = count_label
    weights = torch.tensor([y/(x+y), x/(x+y)]).cuda()
    criterion = RCAMLoss(weight=weights)
    model = getModel(backbone='resnet50', num_class=config['num_class'])
    model = model.cuda()
    model.load_state_dict(get_saved_model(save_dir='train_result_RCAM50'))
    # from line_profiler import LineProfiler
    # lp = LineProfiler()
    # lp_wrapper = lp(train)
    # lp_wrapper(train_loader, val_loader, model, None, optim.AdamW(lr=3e-4, params=model.parameters()), 0, logger)
    # lp.print_stats()
    lr = 3e-4
    for epoch in range(1000):
        optimizer = optim.AdamW(lr=lr, params=model.parameters())
        train(train_loader, val_loader, model, criterion, optimizer, epoch=epoch, logger=logger)
        lr = lr/1.5


    # import os
    # from tqdm import tqdm
    # import pickle
    # acc_list = [[[0.,0.,0.,0.] for _ in range(3)] for __ in range(26)]
    # val_loader = ImageDataLoader(mode='test', shuffle=False, batch_size=1024, num_workers=5)
    # for weights in tqdm(os.listdir('train_result/state_dict')[1:]):
    #     epoch = int(weights[weights.rfind('epoch')+5:weights.rfind('_')])
    #     iter = int(weights[weights.rfind('_')+1:weights.rfind('.')])
    #     if iter == 6635:
    #         iter = 0
    #     elif iter == 13270:
    #         iter = 1
    #     elif iter == 19904:
    #         iter = 2
    #     model.load_state_dict(torch.load(os.path.join('train_result/state_dict',weights)))
    #     test_loss, test_recall, test_pre, test_acc = test_model(model, val_loader, criterion)
    #     recall = test_recall.avg[1]
    #     pre = test_pre.avg[1]
    #     acc = test_acc.avg
    #     f1 = 2 * recall * pre / (recall + pre)
    #     acc_list[epoch][iter][0] = recall
    #     acc_list[epoch][iter][1] = pre
    #     acc_list[epoch][iter][2] = f1
    #     acc_list[epoch][iter][3] = acc
    #     print(epoch, iter)
    #     print([f1, acc])
    #     with open('acc_test_list', 'wb') as file_obj:
    #         pickle.dump(acc_list, file_obj)




