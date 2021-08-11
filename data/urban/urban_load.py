import os
import sys
import librosa
import torch
import pickle
import pandas as pd
import numpy as np
import DataCollection as dc

def load_dataset(data,labels_dict,size=100000,sr=22050):
    data_folds={}
    label_folds={}
    for f in range(10):
        data_temp = torch.empty(0,1,size)
        label_temp= torch.empty(0,1,size)
        fold = data.loc[data['fold']==f+1]
        classes = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
        for i in range(fold.shape[0]):
            fullpath, class_id = dc.path_class(data,data.slice_file_name[i])
            X, sample_rate = librosa.load(fullpath, res_type='kaiser_fast')
            lab = labels_dict[class_id]
            classes[lab].extend(X)

        for i in range(10):
            datas = torch.tensor(classes[i])[:size].unsqueeze(0).unsqueeze(0)
            data_temp = torch.cat([data_temp,datas])
            labs = torch.zeros(1,1,size)
            labs[:]=i
            label_temp = torch.cat([label_temp,labs])

        data_folds[f]=data_temp
        label_folds[f]=label_temp
        
    return data_folds,label_folds

def save_dict(data_folds,label_folds,size=100000):

    train_data = {}
    test_data = {}
    train_labels = {}
    test_labels = {}

    for cv in range(10):
        train_data[cv] = torch.empty(0,1,size)
        train_labels[cv] = torch.empty(0)
        for f in range(10):
            if f==cv:
                test_data[cv]=data_folds[f]
                test_labels[cv]=label_folds[f].squeeze()
            else:
                train_data[cv] = torch.cat([train_data[cv],data_folds[f]])
                train_labels[cv] = torch.cat([train_labels[cv],label_folds[f].squeeze()])
    for i in range(10):
        train_data[i] = train_data[i].numpy()
        test_data[i] = test_data[i].numpy()
        train_labels[i] = train_labels[i].numpy()
        test_labels[i] = test_labels[i].numpy()
        
    return train_data,test_data,train_labels,test_labels



if __name__ == "__main__":
    
    if not os.path.exists('8hY5ER'):
        os.system('wget https://goo.gl/8hY5ER')
        
    print('Extracting Files...')
    if not os.path.exists('UrbanSound8K'):
        os.system('tar -zxvf 8hY5ER')

    data = pd.read_csv("UrbanSound8K/metadata/UrbanSound8K.csv")

    labels_dict = {'air_conditioner':0,
                   'car_horn':1,
                   'children_playing':2,
                   'dog_bark':3,
                   'drilling':4,
                   'engine_idling':5,
                   'gun_shot':6,
                   'jackhammer':7,
                   'siren':8,
                   'street_music':9}
        

        
    data_folds,label_folds = load_dataset(data,labels_dict)
    train_data,test_data,train_labels,test_labels = save_dict(data_folds,label_folds)
    
    if not os.path.exists('processed'):
        os.mkdir('processed')
    with open(os.path.join('processed', 'train_data.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    with open(os.path.join('processed', 'test_data.pkl'), 'wb') as f:
        pickle.dump(test_data, f)
    with open(os.path.join('processed', 'train_labels.pkl'), 'wb') as f:
        pickle.dump(train_labels, f)
    with open(os.path.join('processed', 'test_labels.pkl'), 'wb') as f:
        pickle.dump(test_labels, f)
        
        
     