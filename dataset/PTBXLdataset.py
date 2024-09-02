import torch
import torch.nn as nn
import torch.utils.data as data
import json
import os
import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle

def np_move_avg_batch(a,n=10,mode="same"):
    for j in range(len(a)):
        for i in range(12):
            a[j,i] = np.convolve(a[j,i], np.ones((n,))/n, mode=mode)
    return a

class PTBXLOridataset(data.Dataset):
    '''
    Work flow: 
    剔除同一个人有Norm和得病(且只有一种病)的数据, 作为测试集Patient_Selected_291.csv, 剩余全部为训练集
    多标签的数据没有剔除
    把csv中的report作为description, 5个superclass: HYP, MI, CD, STTC, NORM

    '''
    def __init__(self, root='', train=True, classifier=False, choose_norm=False):
        '''
        root: 
        description: scp_statements.csv
        '''
        classes = ['HYP', 'MI', 'CD', 'STTC', 'NORM']
        self.root = root
        self.cls_map = {str([classes[id]]): id for id in range(len(classes))}

        self.choose_norm = choose_norm
        self.classifier = classifier
        self.train = train
        
        if train:
            self.description = pd.read_csv(os.path.join(root, 'Train_sclc.csv'))
            self.ecg = np.load(os.path.join(root, 'Train_sclc_X.npy'))
            self.crop_wave = np.load(os.path.join(root, 'Train_sclc_X_crop_wave.npy'))
            self.report = self.description['report'].to_list()
            self.ecg_cls = self.description['detail_superclass'].to_list()
            # choose normal
            if choose_norm:
                abnormal_cls = self.description['detail_superclass']  # use 20 classes
                choose_class = []
                for i in range(len(abnormal_cls)):
                    abnormal = eval(abnormal_cls[i])
                    if len(abnormal)==1 and (abnormal[0] == 'NORM'):
                        choose_class.append(True)
                    else:
                        choose_class.append(False)

                self.ecg_cls_style = self.description['detail_superclass'][choose_class].to_list()
                self.ecg_style = self.ecg[choose_class]
                self.crop_wave_style = self.crop_wave[choose_class]
                self.report_style = list(self.description['report'][choose_class])

        else:
            self.description = pd.read_csv(os.path.join(root, 'Patient_Selected_291_sclc.csv'))
            self.ecg = np.load(os.path.join(root, 'Patient_Selected_291_sclc_X.npy')) # 732 条
            self.ecg_cls = self.description['detail_superclass'].to_list()
            self.report = self.description['report'].to_list()
            self.crop_wave = np.load(os.path.join(root, 'Patient_Selected_291_sclc_X_crop_wave.npy'))

            self.description_style = pd.read_csv(os.path.join(root, 'Train_sclc.csv'))
            # choose ill class
            abnormal_cls = self.description_style['detail_superclass']  # use 20 classes
            choose_class = []
            for i in range(len(abnormal_cls)):
                abnormal = eval(abnormal_cls[i])
                if len(abnormal)==1 and (abnormal[0] != 'NORM'):
                    choose_class.append(True)
                else:
                    choose_class.append(False)

            self.ecg_cls_style = list(self.description_style['detail_superclass'][choose_class])
            self.ecg_style = np.load(os.path.join(root, 'Train_sclc_X.npy'))[choose_class]
            self.crop_wave_style = np.load(os.path.join(root, 'Train_sclc_X_crop_wave.npy'))[choose_class]
            self.report_style = list(self.description_style['report'][choose_class])


    def sample_weight(self, path):
        with open(os.path.join(self.root, path), 'r') as f:
            weight_id = json.load(f)
        weight = []
        for c in self.ecg_cls:
            ctype = eval(c)[0]
            weight.append(weight_id[ctype])
        
        return weight

    def __len__(self):
        return len(self.ecg)
    
    def to_one_hot(self, idx):
        out = np.zeros(len(self.cls_map), dtype=float)
        out[idx] = 1
        return out
    
    def __getitem__(self, item):
        # random 取一个normal的
        
        if self.train:
            if self.choose_norm:
                ind = (item + random.randint(1, len(self.ecg_style)-1)) % len(self.ecg_style)
                ecg_1 = np.concatenate([np.transpose(self.ecg[item]), self.crop_wave[item]], axis=0)
                descript_1 = self.report[item]
                cls_1 = self.cls_map[self.ecg_cls[item]]
                ecg_2 = np.concatenate([np.transpose(self.ecg_style[ind]), self.crop_wave_style[ind]], axis=0)
                descript_2 = self.report_style[ind]
                cls_2 = self.cls_map[self.ecg_cls_style[ind]]
            else:
                ind = (item + random.randint(1, len(self.ecg)-1)) % len(self.ecg)
                ecg_1 = np.concatenate([np.transpose(self.ecg[item]), self.crop_wave[item]], axis=0)
                descript_1 = self.report[item]
                cls_1 = self.cls_map[self.ecg_cls[item]]
                ecg_2 = np.concatenate([np.transpose(self.ecg[ind]), self.crop_wave[ind]], axis=0)
                descript_2 = self.report[ind]
                cls_2 = self.cls_map[self.ecg_cls[ind]]
           
        else:
            ecg_1 = np.concatenate([np.transpose(self.ecg_style[item]), self.crop_wave_style[item]], axis=0)
            descript_1 = self.report_style[item]
            cls_1 = self.cls_map[self.ecg_cls_style[item]]
            ind = random.randint(0, len(self.ecg)-1)
            ecg_2 = np.concatenate([np.transpose(self.ecg[ind]), self.crop_wave[ind]], axis=0)
            descript_2 = self.report[ind]
            cls_2 = self.cls_map[self.ecg_cls[ind]]

        return ecg_1, descript_1, cls_1, ecg_2, descript_2, cls_2


class PTBXLTestClsdataset(data.Dataset):
    '''
    合并生成的和真实的，一起分类，一起测试
    1、用所有正常的, 根据训练集的weight，生成相同数量的ecg
    2、用不正常的，生成很多正常的部分
    dataset只负责导入npy和cls类别，测试数据始终为291个人的ecg，但选择是否有病
    '''
    def __init__(self, root, train=True, train_lis=[], choose=['NORM',], avg=None):
        '''
        root: 
        description: scp_statements.csv
        '''
        self.classes = ['HYP', 'MI', 'CD', 'STTC', 'NORM']
        self.root = root
        self.cls_map = {self.classes[id]: id for id in range(len(self.classes))}
        self.train = train
        self.avg = avg
        if train:
            self.ecg = []
            self.ecg_cls = []
            self.patient_id = []
            self.report = []
            self.description = []
            for name in train_lis:
                description = pd.read_csv(name+'.csv')
                ecg = np.load(name+'.npy')
                if name[-3:] == 'ecg':
                    print("move avg!")
                    ecg = np_move_avg_batch(ecg)
                if ecg.shape[1] == 5000:
                    ecg = ecg.transpose(0, 2, 1)
                    ecg = ecg[:, :, 452:4548]
                ecg_cls = description['detail_superclass'].to_list()
                self.patient_id.extend(description['patient_id'].to_list())
                self.ecg_cls.extend(ecg_cls)
                self.ecg.append(ecg)
                self.report.extend(description['report'].to_list())
                self.description.append(description)
            
            self.description = pd.concat(self.description)
            self.ecg = np.concatenate(self.ecg, axis=0)
            print(len(self.patient_id))
            if choose:
                choose_class = []
                for i in range(len(self.ecg_cls)):
                    abnormal = eval(self.ecg_cls[i])
                    if len(abnormal)==1 and (abnormal[0] in choose): # choose abnormal
                    # if len(abnormal)==1 and (abnormal[0] == self.classes[4]): # choose normal
                        choose_class.append(True) 
                    else:
                        choose_class.append(False)
                self.ecg = self.ecg[choose_class]
                choose_item = np.argwhere(choose_class == True).astype(int)
                # self.description = description[choose_class]
                self.ecg_cls = self.description[choose_class]['detail_superclass'].to_list()
                self.report = self.description[choose_class]['report'].to_list()
                self.patient_id = self.description[choose_class]['patient_id'].to_list()
                print(len(self.patient_id))

        else:
            description = pd.read_csv(os.path.join(root, 'Patient_Selected_291_sclc.csv'))
            ecg = np.load(os.path.join(root, 'Patient_Selected_291_sclc_X.npy')) # 732 条
            ecg_cls = description['detail_superclass'].to_list()

            choose_class = []
            for i in range(len(ecg_cls)):
                abnormal = eval(ecg_cls[i])
                if len(abnormal)==1 and (abnormal[0] in choose): # choose abnormal
                # if len(abnormal)==1 and (abnormal[0] == self.classes[4]): # choose normal
                    choose_class.append(True) 
                else:
                    choose_class.append(False)

            self.ecg = ecg[choose_class].transpose(0, 2, 1)
            self.description = description[choose_class]
            self.ecg_cls = description[choose_class]['detail_superclass'].to_list()
            self.report = description[choose_class]['report'].to_list()
            self.patient_id = description[choose_class]['patient_id'].to_list()
            print(len(self.patient_id))


    def sample_weight(self):
        weight_id_t = {self.classes[id]: 0 for id in range(len(self.classes))}
        # abnormal_cls = list(Y['detail_class'])
        for i in self.ecg_cls:
            if type(i) == str:
                if len(eval(i)) == 0:
                    continue
                cl = eval(i)[0]  # always use the first cls for weight sample
                weight_id_t[cl] += 1
            else:
                weight_id_t[self.classes[i]] += 1

        weight_id = {}
        for key, value in weight_id_t.items():
            percent = value/sum(weight_id_t.values())
            weight_id[key] = 1/percent

        weight = []
        for c in self.ecg_cls:
            if type(c) == str:
                ctype = eval(c)[0]
                weight.append(weight_id[ctype])
            else:
                ctype = self.classes[c]
                weight.append(weight_id[ctype])
        
        return weight

    def __len__(self):
        return len(self.ecg)
    
    def to_one_hot(self, idx):
        out = np.zeros(len(self.cls_map), dtype=float)
        out[idx] = 1
        return out
    
    def __getitem__(self, item):
        ecg = self.ecg[item]
        if self.avg:
            ecg = self.avg(ecg)
        des = self.report[item]
        if type(self.ecg_cls[item]) == str:
            clss = self.cls_map[eval(self.ecg_cls[item])[0]]
        else:
            clss = self.ecg_cls[item]
        p_id = int(self.patient_id[item])
        return ecg, clss, p_id, des
        

def collect_fn_ori(batch):
    # abnormal_ecg & normal_ecg
    ab_ecg = torch.stack([torch.tensor(item[0]) for item in batch], dim=0)
    nor_ecg = torch.stack([torch.tensor(item[3]) for item in batch], dim=0)
    descrip = [item[1] for item in batch]
    ab_type = torch.tensor(np.array([item[2] for item in batch]))
    norm_type = torch.tensor(np.array([item[5] for item in batch]))
    norm_descrip = [item[4] for item in batch]
    return ab_ecg, descrip, ab_type, nor_ecg, norm_descrip, norm_type


if __name__ == "__main__":
    dataset = PTBXLTestClsdataset('ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/reprepared/', 
                                      True, choose=['HYP', 'MI', 'CD', 'STTC', 'NORM'],
                                      train_lis=['ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/reprepared/Patient_Select_145_sclc_X_half1'])
    loader = data.DataLoader(dataset=dataset,batch_size=4,num_workers=1,shuffle=True)
    for idx, batch in enumerate(dataset):
        print(batch)
    
   
