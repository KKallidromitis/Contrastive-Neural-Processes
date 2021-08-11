import os
import numpy as np
import pandas as pd
import torch
import pickle
import random
random.seed(41)
'''
This code is used to generate the data and labels for the 1st experiment of the IMS bearing Dataset:

https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan


See below for label info
'''

def gen_labels_experiment_1(directory_files,number_files):
    
    classes_dict = {'early':0,'normal':1,'suspect':2,'imminent failure':3,'inner race failure':4,
                    'rolling element failure':5,'stage 2 failure':6}
    
    exp_labels = np.zeros((number_files,4))
    
    bearing_files = {0:['2003.10.23.09.14.13','2003.11.08.12.11.44','2003.11.19.21.06.07','2003.11.24.20.47.32'],
                1:['2003.11.01.21.41.44','2003.11.24.01.01.24','2003.11.25.10.47.32'],
                2:['2003.11.01.21.41.44','2003.11.22.09.16.56','2003.11.25.10.47.32'],
                3:['2003.10.29.21.39.46','2003.11.15.05.08.46','2003.11.18.19.12.30','2003.11.22.17.36.56']}
    
    bearing_states = {0:['early','suspect','normal','suspect','imminent failure'],
                1:['early','normal','suspect','imminent failure'],
                2:['early','normal','suspect','inner race failure'],
                3:['early','normal','suspect','rolling element failure','stage 2 failure']}
    
    
    for b_id in range(4):
        index = [np.where(directory_files==bearing_files[b_id][i])[0][0] for i in range(len(bearing_files[b_id]))]
        
        for i in range(len(bearing_states[b_id])):
            
            if i==0:
                exp_labels[0:(index[i]+1),b_id]=classes_dict[bearing_states[b_id][i]]
                
            elif i==len(index):
                exp_labels[(index[i-1]+1):,b_id]=classes_dict[bearing_states[b_id][i]]
            else:
                exp_labels[(index[i-1]+1):(index[i]+1),b_id]=classes_dict[bearing_states[b_id][i]]
    
    labels = np.zeros((4,20480*number_files))
    
    for i in range(4):
        for j in range(number_files):
            labels[i][(j*20480):((j+1)*20480)] = [exp_labels[j][i]]*20480
    
    return labels

def normalize(train_data, test_data):
    """ Calculate the mean and std of each feature from the training set
    """
    feature_means = np.mean(train_data, axis=(0, 2))
    feature_std = np.std(train_data, axis=(0, 2))
    train_data_n = train_data - feature_means[np.newaxis, :, np.newaxis] / \
                    np.where(feature_std == 0, 1, feature_std)[np.newaxis, :, np.newaxis]
    test_data_n = test_data - feature_means[np.newaxis, :, np.newaxis] /\
                    np.where(feature_std == 0, 1, feature_std)[np.newaxis, :, np.newaxis]
    return train_data_n, test_data_n

def load_experiment(load_path,exp='1st_test'):
    
    load_path = load_path+exp+'/'
    files = np.sort(os.listdir(load_path))
    number_files=len(files)
    
    if exp=='1st_test':
        labels = gen_labels_experiment_1(files,number_files)
    
    load_all = np.zeros((8,20480*len(files)))
    data = np.zeros((4,2, 20480*len(files)))

    for i in range(len(files)): #len(files)
        temp = pd.read_csv(load_path+files[i],sep='\t',header=None).to_numpy().T
        load_all[:,(i*20480):(i+1)*20480] = temp

        print('Loading ',i,end="\r")

    for i in range(4):
        data[i] = load_all[[2*i,2*i+1],:]
    return torch.tensor(data),torch.tensor(labels)

def preprocess(load_path='./raw/',save_path='processed',experiments=['1st_test']):

    for e in experiments:
        data,labels = load_experiment(load_path,exp=e)
        
        data = data[3]
        labels = labels[3]
        class_names = np.sort(np.unique(labels))
        for i in range(int(5)):
            labels[labels == class_names[i]] = i

        data=data.unsqueeze(0)
        labels=labels.unsqueeze(0)
    
        T = data.shape[-1]
        data = np.concatenate(np.split(data[ :, :T // 5 * 5], 80, -1), 0)    
        labels = np.concatenate(np.split(labels[ :, :T // 5 * 5], 80, -1), 0)

        shuffled_inds = list(range(len(labels)))

        random.shuffle(shuffled_inds)
        data=data[shuffled_inds]
        labels=labels[shuffled_inds]

        n_train = int(0.8*len(labels))
        train_data = data[:n_train]
        test_data = data[n_train:]

        train_labels = labels[:n_train]
        test_labels = labels[n_train:]

        train_data, test_data = normalize(train_data, test_data)

        if e=='1st_test':
            save_var = 't1'
          
        print(train_data.shape,train_labels.shape)
        print(test_data.shape,test_labels.shape)
        
        with open(os.path.join(save_path,'train_data_'+save_var+'.pkl'), 'wb') as f:
            pickle.dump(train_data, f)
        with open(os.path.join(save_path,'train_labels_'+save_var+'.pkl'), 'wb') as f:
            pickle.dump(train_labels, f)
        with open(os.path.join(save_path,'test_data_'+save_var+'.pkl'), 'wb') as f:
            pickle.dump(test_data, f)
        with open(os.path.join(save_path,'test_labels_'+save_var+'.pkl'), 'wb') as f:
            pickle.dump(test_labels, f)

        print('Done Saving Files Experiment ', e)
        
    return



if __name__ == "__main__":
    try: 
        os.mkdir('./raw')  
    except OSError as error:  
        print(error)
    try: 
        os.mkdir('./processed')  
    except OSError as error:  
        print(error)
        
    os.chdir("./raw")
    
    if not os.path.exists('IMS.7z'):
        os.system('wget https://ti.arc.nasa.gov/m/project/prognostic-repository/IMS.7z')
        
    print('Extracting Files...')
    if not os.path.exists('1st_test.rar'):
        os.system('7za x IMS.7z')
        
    if not os.path.exists('1st_test'):
        os.system('unrar x 1st_test.rar')
        
    os.chdir("../")
    preprocess()


'''
Labels:

Bearing 0
early: 2003.10.22.12.06.24 - 2003.10.23.09.14.13
suspect: 2003.10.23.09.24.13 - 2003.11.08.12.11.44 (bearing 1 was in suspicious health from the beginning, but showed some self-healing effects)
normal: 2003.11.08.12.21.44 - 2003.11.19.21.06.07
suspect: 2003.11.19.21.16.07 - 2003.11.24.20.47.32
imminent failure: 2003.11.24.20.57.32 - 2003.11.25.23.39.56

Bearing 1
early: 2003.10.22.12.06.24 - 2003.11.01.21.41.44
normal: 2003.11.01.21.51.44 - 2003.11.24.01.01.24
suspect: 2003.11.24.01.11.24 - 2003.11.25.10.47.32
imminent failure: 2003.11.25.10.57.32 - 2003.11.25.23.39.56

Bearing 2
early: 2003.10.22.12.06.24 - 2003.11.01.21.41.44
normal: 2003.11.01.21.51.44 - 2003.11.22.09.16.56
suspect: 2003.11.22.09.26.56 - 2003.11.25.10.47.32
Inner race failure: 2003.11.25.10.57.32 - 2003.11.25.23.39.56

Bearing 3
early: 2003.10.22.12.06.24 - 2003.10.29.21.39.46
normal: 2003.10.29.21.49.46 - 2003.11.15.05.08.46
suspect: 2003.11.15.05.18.46 - 2003.11.18.19.12.30
Rolling element failure: 2003.11.19.09.06.09 - 2003.11.22.17.36.56
Stage 2 failure: 2003.11.22.17.46.56 - 2003.11.25.23.39.56




Source: http://mkalikatzarakis.eu/wp-content/uploads/2018/12/IMS_dset.html

'''