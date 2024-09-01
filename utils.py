import torch
import numpy as np
import time,os
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
import matplotlib.pyplot as plt


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def to_one_hot(idx):
    out = np.zeros(6, dtype=float)
    out[idx] = 1
    return out
    
#计算F1score
def calc_f1_acc(y_true, y_pre, threshold=0.5):
    y_true = y_true.cpu().detach().argmax(dim=1).numpy()
    y_pre = y_pre.cpu().detach().argmax(dim=1).numpy()
    return f1_score(y_true, y_pre, average='macro'), sum((y_true==y_pre))/len(y_pre)

def calc_f1_acc_one_hot(y_true, y_pre, threshold=0.5):
    # y_true = np.array([to_one_hot(i) for i in y_true])
    y_true = y_true.cpu().numpy()
    y_pre = y_pre.cpu().detach().argmax(dim=1).numpy()
    return f1_score(y_true, y_pre, average='macro'), sum((y_true==y_pre))/len(y_pre)

def calc_f1_acc_one_hot_pid(y_true, y_pre, p_id, pre, target, threshold=0.5):
    # y_true = np.array([to_one_hot(i) for i in y_true])
    y_true = y_true.cpu().numpy()
    y_pre = y_pre.cpu().detach().numpy()
    p_id = p_id.cpu().numpy()
    for i in range(len(p_id)):
        if p_id[i] in pre.keys():
            pre[p_id[i]].append(y_pre[i])
            target[p_id[i]].append(y_true[i])
        else:
            pre[p_id[i]] = []
            target[p_id[i]] = []
            pre[p_id[i]].append(y_pre[i])
            target[p_id[i]].append(y_true[i])
    y_pre = y_pre.argmax(axis=1)
    return f1_score(y_true, y_pre, average='macro'), sum((y_true==y_pre))/len(y_pre)

def person_avg_acc_f1(person_pre, person_target):
    person_acc = {}
    person_f1 = {}
    for k in person_pre.keys():
        pre = np.array(person_pre[k]) 
        target = np.array(person_target[k])
        person_acc[k] = sum(pre == target)/len(pre)
        person_pre[k] = f1_score(target, pre, average='macro')
    
    avg_acc = sum(person_acc.values())/len(person_acc.values())
    avg_f1 = sum(person_f1.values())/len(person_f1.values())
    
    return avg_f1, avg_acc

def calc_acc(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return (y_true==y_pre)/len(y_pre)


#打印时间
def print_time_cost(since):
    time_elapsed = time.time() - since
    return '{:.0f}m{:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)


# 调整学习率
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

#多标签使用类别权重
class WeightedMultilabel(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super(WeightedMultilabel, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')
        self.weights = weights

    def forward(self, outputs, targets):
        loss = self.cerition(outputs, targets)
        return (loss * self.weights).mean()
    

def save_ecg_image(ecg_edit, save_dir):
    '''
    ecg_edit: (b, lead, 5000)
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    x = ecg_edit.cpu().detach().numpy()
    titles = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    for i in range(len(x)):
        ecg = x[i]
        plt.rcParams['figure.figsize'] = (40.0, 40.0)
        plt.rcParams["axes.grid"] = True
        plt.rcParams["grid.linestyle"] = (0.1,0.1)
        plt.figure()
        for index in range(12):
            plt.subplot(6,2,index+1)
            plt.plot(ecg[index, 500:2000], linewidth=5)
            
            # plt.yticks(np.arange(np.min(ecg[:,index]), np.max(ecg[:,index]), 0.1))
            plt.gca()
            plt.title(titles[index])
            # plt.axis('off')

        plt.savefig(os.path.join(save_dir, f'editecg-{i}.jpg'))


def freeze_model(model):
    """
    Freeze all the parameters in the given PyTorch model.

    Parameters:
    model (nn.Module): A PyTorch model whose parameters need to be frozen.
    """
    for param in model.parameters():
        param.requires_grad = False



