import torch, time, os
import utils
import numpy as np
import pandas as pd
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from model.classifier import *
from dataset.PTBXLdataset import PTBXLTestClsdataset
from config.config import config
from sklearn.metrics import f1_score, roc_auc_score
from model.Inception import GoogLeNet
import scipy.stats as stats
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(41)  
torch.cuda.manual_seed(41) 

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(0)
def softmax(x):

    max = np.max(
        x, axis=1, keepdims=True
    )  # returns max of each row and keeps same dims
    e_x = np.exp(x - max)  # subtracts each row with its max value
    sum = np.sum(
        e_x, axis=1, keepdims=True
    )  # returns sum of each row and keeps same dims
    f_x = e_x / sum
    return f_x

def one_hot(x):
    hot = np.zeros((x.shape[0], 5))
    for i in range(len(x)):
        hot[i][x[i]]=1
    return hot

def print_log(text, path):
    print(text)
    with open(os.path.join(path, 'log.txt'), 'a') as f:
        f.write(text+"\n")

def save_ckpt(state, is_best, model_save_dir):
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    current_w = os.path.join(model_save_dir, config.current_w)
    best_w = os.path.join(model_save_dir, config.best_w)
    torch.save(state, current_w)
    if is_best:
        torch.save(state, best_w)

def train_epoch(model, optimizer, criterion, train_dataloader, show_interval=10, save_dir='./'):
    model.train()
    f1_meter, acc_meter, loss_meter, it_count = 0, 0, 0, 0
    for batch in train_dataloader:
        inputs = batch[0].to(device).float()
        target = batch[1].to(device)
        inputs = inputs[:, :12]
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        f1, acc = utils.calc_f1_acc_one_hot(target, torch.sigmoid(output))
        f1_meter += f1
        acc_meter += acc
        if it_count != 0 and it_count % show_interval == 0:
            print_log("Iter %d,loss:%.3e f1:%.3f acc:%.3f" % (it_count, loss.item(), f1, acc), save_dir)
    return loss_meter / it_count, f1_meter / it_count, acc_meter / it_count


def val_epoch(model, criterion, val_dataloader, threshold=0.5):
    model.eval()
    f1_meter, acc_meter, loss_meter, it_count = 0, 0, 0, 0
    with torch.no_grad():
        for batch in val_dataloader:
            inputs = batch[0].to(device).float()
            inputs = inputs[:, :12]
            target = batch[1].to(device)
            output = model(inputs)
            loss = criterion(output, target)
            loss_meter += loss.item()
            it_count += 1
            output = torch.sigmoid(output)
            f1, acc = utils.calc_f1_acc_one_hot(target, output, threshold)
            f1_meter += f1
            acc_meter += acc
    
    return loss_meter / it_count, f1_meter / it_count, acc_meter / it_count


def val_p_epoch(model, criterion, val_dataloader, best_acc, threshold=0.5, model_save_dir='./'):
    model.eval()
    f1_meter, acc_meter, loss_meter, it_count = 0, 0, 0, 0
    auc_meter = 0
    person_pre = {}
    person_target = {}
    with torch.no_grad():
        for batch in val_dataloader:
            inputs = batch[0].to(device).float()
            # inputs = inputs.transpose(1,2)
            inputs = inputs[:, :12]
            p_id = batch[2].to(device)
            target = batch[1].to(device)
            output = model(inputs)
            loss = criterion(output, target)
            loss_meter += loss.item()
            it_count += 1
            output = torch.sigmoid(output)
            f1, acc = utils.calc_f1_acc_one_hot_pid(target, output, p_id, person_pre, person_target, threshold)
            f1_meter += f1
            acc_meter += acc
            

    person_acc = {}
    person_f1 = {}
    pre_lis = []
    target_lis = []
    for k in person_pre.keys():
        pre = np.array(person_pre[k]) 
        target = np.array(person_target[k])
        person_acc[k] = sum(pre.argmax(axis=1) == target)/len(pre)
        person_f1[k] = f1_score(target, pre.argmax(axis=1), average='macro')
        pre_lis.append(pre)
        target_lis.append(target)
    print(one_hot(np.concatenate(target_lis)))
    try:
        total_auc = roc_auc_score(one_hot(np.concatenate(target_lis)), softmax(np.concatenate(pre_lis)), multi_class='ovr')
    except ValueError:
        total_auc = 0
    corr, p = stats.spearmanr(np.concatenate(target_lis), np.concatenate(pre_lis).argmax(axis=1))
    t_f1 = f1_score(np.concatenate(target_lis), np.concatenate(pre_lis).argmax(axis=1), average='macro')
    t_acc = sum(np.concatenate(pre_lis).argmax(axis=1) == np.concatenate(target_lis))/len(np.concatenate(pre_lis).argmax(axis=1))

    avg_acc = sum(person_acc.values())/len(person_acc.values())
    avg_f1 = sum(person_f1.values())/len(person_f1.values())
    if avg_acc > best_acc:
        with open(os.path.join(model_save_dir,'Person_acc_f1.csv'),'w') as f:
            f.write('patient_id,Acc,F1,num_item\n')
            for p_id in person_acc.keys():
                f.write(f"{p_id},{person_acc[p_id]},{person_f1[p_id]},{len(person_target[p_id])}\n")

            f.write(f"T{len(person_acc.keys())},{t_acc},{t_f1},{total_auc}, {corr}, {p},{loss_meter / it_count}\n")

    print_log(f"Total f1:{f1_meter / it_count} acc: {acc_meter / it_count} auc: {total_auc} corr: {corr} p:{p}\n Person acc: {avg_acc} f1: {avg_f1}\n", model_save_dir)
    # return loss_meter / it_count, f1_meter / it_count, acc_meter / it_count
    return loss_meter / it_count, avg_f1, avg_acc, f1_meter / it_count, acc_meter / it_count

def train(args):
    # 模型保存文件夹
    model_save_dir = '%s/%s/%s' % (config.ckpt, args.ex, time.strftime("%Y%m%d%H%M"))
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
        os.makedirs(model_save_dir+'/half1/')
        os.makedirs(model_save_dir+'/total/')

    train_dataset = PTBXLTestClsdataset('/data2/huyaojun/PublicDataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/reprepared/',
                                         True, choose=None, 
                                         train_lis=[#'work_dir/Gen_Wcgan/202401130438/ecg',
                                             'work_dir/Gen_half1/202312211352/ecg',
                                             #'work_dir/home/work_dir/Gen_05Nowave200/202403192258/ecg',
                                            #'work_dir/home/work_dir/Gen_05Nowave100/202403192256/ecg',
                                             '/data2/huyaojun/PublicDataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/reprepared/Train_sclc_X'
                                             ])
    train_weight = train_dataset.sample_weight()
    train_sampler = WeightedRandomSampler(train_weight, len(train_weight))
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler,
                                num_workers=4, drop_last=True)

    val_dataset = PTBXLTestClsdataset('/data2/huyaojun/PublicDataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/reprepared/', 
                                      True, choose=['HYP', 'MI', 'CD', 'STTC', 'NORM'],
                                      train_lis=['/data2/huyaojun/PublicDataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/reprepared/Patient_Select_145_sclc_X_half2'])  # choose abnormal
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True,
                                num_workers=4, drop_last=True)
    val_dataset1 = PTBXLTestClsdataset('/data2/huyaojun/PublicDataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/reprepared/', 
                                      True, choose=['HYP', 'MI', 'CD', 'STTC', 'NORM'],
                                      train_lis=['/data2/huyaojun/PublicDataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/reprepared/Patient_Select_145_sclc_X_half1'])  # choose abnormal

    val_dataloader1 = DataLoader(val_dataset1, batch_size=config.batch_size, shuffle=True,
                                num_workers=4, drop_last=True)
    val_datasett = PTBXLTestClsdataset('/data2/huyaojun/PublicDataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/reprepared/', 
                                      False, choose=['HYP', 'MI', 'CD', 'STTC', 'NORM'],
                                      train_lis=['/data2/huyaojun/PublicDataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/reprepared/Patient_Select_291_sclc_X'])  # choose abnormal
    
    val_dataloadert = DataLoader(val_datasett, batch_size=config.batch_size, shuffle=True,
                                num_workers=4, drop_last=True)
    print_log(f"train_datasize {len(train_dataset)} val_datasize {len(val_dataset)} val_datasize1 {len(val_dataset1)} val_datasizet {len(val_datasett)}", model_save_dir)
    print(len(train_dataset.cls_map))
    model = resnet34(num_classes=len(train_dataset.cls_map))
    # model = resnet101(num_classes=len(train_dataset.cls_map))
    # model = GoogLeNet(len(train_dataset.cls_map), 12)
    # if args.ckpt and not args.resume:
    #     state = torch.load(args.ckpt, map_location='cpu')
    #     model.load_state_dict(state['state_dict'])
    #     print('train with pretrained weight val_f1', state['f1'])
    model = model.to(device)
    # data
    # optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()


    # if args.ex: model_save_dir += args.ex
    best_f1 = -1
    best_acc = -1
    best_f11 = -1
    best_acc1 = -1
    best_f12 = -1
    best_acc2 = -1
    lr = config.lr
    start_epoch = 1
    stage = 1

    if args.resume:
        if os.path.exists(args.ckpt): 
            model_save_dir = args.ckpt
            current_w = torch.load(os.path.join(model_save_dir, config.current_w))
            best_w = torch.load(os.path.join(model_save_dir, config.best_w))
            best_f1 = best_w['f1']
            start_epoch = current_w['epoch'] + 1
            lr = current_w['lr']
            stage = current_w['stage']
            model.load_state_dict(current_w['state_dict'])

            if start_epoch - 1 in config.stage_epoch:
                stage += 1
                lr /= config.lr_decay
                utils.adjust_learning_rate(optimizer, lr)
                model.load_state_dict(best_w['state_dict'])
            print_log("=> loaded checkpoint (epoch {})".format(start_epoch - 1), model_save_dir)

    logger = Logger(logdir=model_save_dir, flush_secs=2)

    for epoch in range(start_epoch, config.max_epoch + 1):
        since = time.time()
        train_loss, train_f1, train_acc = train_epoch(model, optimizer, criterion, train_dataloader, show_interval=100, save_dir=model_save_dir)
        val_loss, val_f1, val_acc, t_f1, t_acc = val_p_epoch(model, criterion, val_dataloader, best_acc, model_save_dir=model_save_dir)
        val_loss1, val_f11, val_acc1, t_f11, t_acc1 = val_p_epoch(model, criterion, val_dataloader1, best_acc1, model_save_dir=model_save_dir+'/half1/')
        val_loss2, val_f12, val_acc2, t_f12, t_acc2 = val_p_epoch(model, criterion, val_dataloadert, best_acc2, model_save_dir=model_save_dir+'/total/')
        
        print_log('#epoch:%02d stage:%d train_loss:%.3e train_f1:%.3f  train_acc:%.3f val_loss:%0.3e val_f1:%.3f val_acc:%.3f time:%s\n'
              % (epoch, stage, train_loss, train_f1, train_acc, val_loss, val_f1, val_acc, utils.print_time_cost(since)), model_save_dir)
        logger.log_value('train_loss', train_loss, step=epoch)
        logger.log_value('train_f1', train_f1, step=epoch)
        logger.log_value('train_acc', train_acc, step=epoch)
        logger.log_value('val_loss', val_loss, step=epoch)
        logger.log_value('val_f1', val_f1, step=epoch)
        logger.log_value('val_acc', val_acc, step=epoch)
        logger.log_value('val_t_f1', t_f1, step=epoch)
        logger.log_value('val_t_acc', t_acc, step=epoch)
        logger.log_value('val_loss1', val_loss1, step=epoch)
        logger.log_value('val_f11', val_f11, step=epoch)
        logger.log_value('val_acc1', val_acc1, step=epoch)
        logger.log_value('val_t_f11', t_f11, step=epoch)
        logger.log_value('val_t_acc1', t_acc1, step=epoch)
        logger.log_value('val_loss2', val_loss1, step=epoch)
        logger.log_value('val_f12', val_f11, step=epoch)
        logger.log_value('val_acc2', val_acc1, step=epoch)
        logger.log_value('val_t_f12', t_f11, step=epoch)
        logger.log_value('val_t_acc2', t_acc1, step=epoch)
        state = {"state_dict": model.state_dict(), "epoch": epoch, "loss": val_loss, 'f1': val_f1, 'acc': val_acc, 'lr': lr,
                 'stage': stage, 't_f1': t_f1, 't_acc': t_acc}
        
        save_ckpt(state, best_acc < val_acc, model_save_dir)
        save_ckpt(state, best_acc1 < val_acc1, model_save_dir+'/half1/')
        save_ckpt(state, best_acc2 < val_acc2, model_save_dir+'/total/')
        best_f1 = max(best_f1, val_f1)
        best_acc = max(best_acc, val_acc)
        print_log(f"Best f1: {best_f1}, Best Acc: {best_acc}", model_save_dir)
        best_f11 = max(best_f11, val_f11)
        best_acc1 = max(best_acc1, val_acc1)
        print_log(f"Best f1: {best_f11}, Best Acc: {best_acc1}", model_save_dir+'/half1/')
        best_f12 = max(best_f12, val_f12)
        best_acc2 = max(best_acc2, val_acc2)
        print_log(f"Best f1: {best_f12}, Best Acc: {best_acc2}", model_save_dir+'/total/')
        if epoch in config.stage_epoch:
            stage += 1
            lr /= config.lr_decay
            best_w = os.path.join(model_save_dir, config.best_w)
            model.load_state_dict(torch.load(best_w)['state_dict'])
            print_log("**************step into stage%02d lr %.3ef ************" % (stage, lr), model_save_dir)
            utils.adjust_learning_rate(optimizer, lr)

# #用于测试加载模型
def val(args):
    model_save_dir = '%s/%s/%s' % (config.ckpt, args.ex, time.strftime("%Y%m%d%H%M"))
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir+'/total/')
    list_threhold = [0.5]
    # model = resnet34(num_classes=5)
    model = resnet101(num_classes=5)
    model.load_state_dict(torch.load("ckpt2/ResNet101/202312300028/best_w.pth",
                                                    map_location='cpu')['state_dict'])
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    val_datasett = PTBXLTestClsdataset('/data2/huyaojun/PublicDataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/reprepared/',
                                         True, choose=None, 
                                         train_lis=['work_dir/Gen/202312291712/ecg',])
                                            #  '/data2/huyaojun/PublicDataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/reprepared/Patient_Select_145_sclc_X_half1',
                                            #  '/data2/huyaojun/PublicDataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/reprepared/Patient_Select_145_sclc_X_half2',
                                            #  '/data2/huyaojun/PublicDataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/reprepared/Train_sclc_X'])
    # val_datasett = PTBXLTestClsdataset('/data2/huyaojun/PublicDataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/reprepared/', 
    #                                   True, choose=['HYP', 'MI', 'CD', 'STTC', 'NORM'],
    #                                   train_lis=['/data2/huyaojun/PublicDataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/reprepared/Patient_Select_145_sclc_X_half2'])  # choose abnormal
    val_dataloadert = DataLoader(val_datasett, batch_size=config.batch_size, 
                                num_workers=4, drop_last=True)
    best_acc2 = 0
    best_f12 = 0
    val_loss2, val_f12, val_acc2, t_f12, t_acc2 = val_p_epoch(model, criterion, val_dataloadert, best_acc2, model_save_dir=model_save_dir+'/total/')
    best_f12 = max(best_f12, val_f12)
    best_acc2 = max(best_acc2, val_acc2)
    print_log(f"Best f1: {best_f12}, Best Acc: {best_acc2} {t_f12} {t_acc2}", model_save_dir+'/total/')



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="classifier", help="the path of model weight file")
    parser.add_argument("--ex", type=str, default='baseline', help="experience name")
    parser.add_argument("--resume", action='store_true', default=False)
    args = parser.parse_args()
    
    train(args)
    # val(args)
