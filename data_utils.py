import os
import pickle
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib import gridspec
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import torch.nn as nn
from torch.utils import data
import torch
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from torchvision import datasets
from models import select_encoder
from sklearn.metrics import silhouette_score,davies_bouldin_score
from models import Conv1DEncoder,RnnEncoder
from audiomentations import Compose, AddBackgroundNoise, AddGaussianNoise, FrequencyMask, TimeMask, TimeStretch
#from torch_audiomentations import *

def shuffle(data,labels):
    inds = list(range(len(labels)))
    random.shuffle(inds)
    labels = labels[inds]
    data = data[inds]

    return data,labels

def load_datasets(data_type,datasets=None,cv=None):
    
    if data_type=='afdb':

        path = './data/afdb/processed/'
        train_data=pickle.load(open(path+'x_train.pkl','rb'))
        train_labels = pickle.load(open(path+'state_train.pkl','rb'))

        test_labels = pickle.load(open(path+'state_test.pkl','rb'))
        test_data = pickle.load(open(path+'x_test.pkl','rb'))
    
    elif data_type=='ims':

        path = './data/ims/processed/'
        train_data = pickle.load(open(path+'train_data_t1.pkl','rb'))
        train_labels = pickle.load(open(path+'train_labels_t1.pkl','rb'))

        test_data = pickle.load(open(path+'test_data_t1.pkl','rb'))
        test_labels = pickle.load(open(path+'test_labels_t1.pkl','rb'))
        
    elif data_type=='urban':

        path = './data/urban/processed/'
        train_data = pickle.load(open(path+'train_data.pkl','rb'))
        train_labels = pickle.load(open(path+'train_labels.pkl','rb'))

        test_data = pickle.load(open(path+'test_data.pkl','rb'))
        test_labels = pickle.load(open(path+'test_labels.pkl','rb'))
        
        train_data = train_data[cv]
        test_data = test_data[cv]
        train_labels = train_labels[cv]
        test_labels = test_labels[cv]

    return train_data,train_labels,test_data,test_labels


def split_series(data,labels,window_size):
    
    if window_size == -1:
        return data.float(),labels
    
    try:
        labels = labels.numpy()
    except:
        labels=labels
        
    T = data.shape[-1]
    x_window = np.split(data[:, :, :window_size * (T // window_size)],(T//window_size), -1)
    x_window = torch.Tensor(np.concatenate(x_window, 0))
    
    if torch.tensor(labels).clone().detach().dim()!=1:
        y_window = np.concatenate(np.split(labels[:, :window_size * (T // window_size)], (T // window_size), -1), 0).astype(int)
        y_window = torch.Tensor(np.array([np.bincount(yy).argmax() for yy in y_window]))
    else:
        y_window = torch.tensor(labels)
        
    return x_window,y_window

def gen_loader(data_type,datasets,n_classes,tr_percentage=0.8,val_percentage=0.2,window_size=2500,batch_size=20,cv=None):
    
    train_data,train_labels,test_data,test_labels = load_datasets(data_type,datasets,cv)
    
    if batch_size<1:
        batch_size = max(1,int(min(len(train_data),len(test_data))*batch_size))
        print('Using batch_size:', batch_size)
        
    train_data,train_labels = split_series(train_data,train_labels,window_size)
    test_data,test_labels = split_series(test_data,test_labels,window_size)
        
    n_train = int(tr_percentage*len(train_labels))
    n_val = int(val_percentage*len(train_labels))

    train_data,train_labels = shuffle(train_data,train_labels)
    test_data,test_labels = shuffle(test_data,test_labels)
    
    trainset = torch.utils.data.TensorDataset(train_data[:n_train], train_labels[:n_train])
    validset = torch.utils.data.TensorDataset(train_data[-n_val:], train_labels[-n_val:])
    testset = torch.utils.data.TensorDataset(test_data, test_labels)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,drop_last=True)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True,drop_last=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True,drop_last=True)
    
    return train_loader,valid_loader,test_loader

def compute_avg(arr):
    cv = len(arr)
    avg_arr = np.sum([arr[i] for i in range(cv)],axis=0)/cv
    return avg_arr

def log_data(name,arr,names):
    obj = {}
    for i in range(len(names)):
        obj[names[i]] = arr[i]

    with open(name+'.pkl', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return

class Plots():        
    def plot_distribution(self,x_test, y_test, encoder_type,encoding_size, window_size, model_type,
                          datasets, data_type,suffix, device, title="", cv=0,parallel=False,augment=100):
        
        checkpoint = torch.load('./results/baselines/%s_%s/%s/encoding_%d_encoder_%d_checkpoint_%d%s.pth.tar' 
                                %(datasets,model_type,data_type,encoding_size,encoder_type, cv,suffix))
        input_size = [x.shape for x in x_test][0][0]
        encoder,_ = select_encoder(device,encoder_type,input_size,encoding_size)
        
        if parallel:
            device_ids = [i for i in range(torch.cuda.device_count())]
            encoder = nn.DataParallel(encoder,device_ids)
        encodings=None
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        print(x_test.shape)
        n_test = len(x_test)
        inds = np.random.randint(0, x_test.shape[-1] - window_size, n_test * augment)
        windows = np.array([x_test[int(i % n_test), :, ind:ind + window_size] for i, ind in enumerate(inds)])
        windows_state = [np.round(np.mean(y_test[i % n_test, ind:ind + window_size], axis=-1)) for i, ind in
                        enumerate(inds)]
        testset = torch.utils.data.TensorDataset(torch.Tensor(windows).to(device))
                                                 
        test_loader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False,drop_last=False)
        
        for x in test_loader:

            enc = encoder(x[0])
            if encodings==None:
                encodings = enc.detach().to('cpu')
            else:
                encodings = torch.cat([encodings,enc.detach().to('cpu')])
                
        tsne = TSNE(n_components=2)
        embedding = tsne.fit_transform(encodings.detach().cpu().numpy())
        df_encoding = pd.DataFrame({"f1": embedding[:, 0], "f2": embedding[:, 1], "state": windows_state})
        figure, ax = plt.subplots(1,1)
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(16, 12)
        figure.tight_layout()
        
        sns.set_style("white")
        sns.scatterplot(x="f1", y="f2", data=df_encoding, hue="state", palette="deep")
        # sns.jointplot(x="f1", y="f2", data=df_encoding, kind="kde", hue='state')
        fig.savefig('./results/baselines/%s_%s/encoding_%s_%s_%s.svg'%(data_type,model_type,data_type,model_type,cv), dpi=300, format='svg', bbox_inches = 'tight', pad_inches = 0.0)
        plt.show()
        
        return

    def plot_acc_loss(self,model,train_accs,test_accs,train_losses,test_losses):
        plt.title(model+' Accuracy vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.plot(train_accs,marker='o',color='b', label="Train")
        plt.plot(test_accs,marker='s',color='r', label="Test")
        plt.legend()
        plt.show()
        

        plt.title(model+' Loss vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.plot(train_losses,marker='o',color='b', label="Train")
        plt.plot(test_losses,marker='s',color='r', label="Test")
        plt.legend()
        plt.show()

        return


    def plot_ablation(self,cv,tr,val_accs_dict,test_accs_dict,test_losses_dict):
        accuracies = []
        losses = []
        print('Label Ablation Results')
        for j in tr:
            ids = [np.argmax(val_accs_dict[j][i]) for i in range(cv)]
            nums_acc = [test_accs_dict[j][i][ids[i]] for i in range(cv)]
            nums_loss = [test_losses_dict[j][i][ids[i]] for i in range(cv)]
            
            print('Label (%)',j,'Test Acuracy:', np.mean(nums_acc),max((np.mean(nums_acc) - min(nums_acc)),(max(nums_acc)-np.mean(nums_acc))))
            accuracies.append(np.mean(nums_acc))
            losses.append(np.mean(nums_loss))
        
        plt.title('Test Accuracy vs Label (%)')
        plt.xlabel('Label (%)')
        plt.ylabel('Test Accuracy (%)')
        plt.plot(tr,accuracies,marker='s',color='r',)                 
        plt.show()
        
        plt.title('Test Loss vs Label (%)')
        plt.xlabel('Label (%)')
        plt.ylabel('Test Loss')
        plt.plot(tr,losses,marker='s',color='r')                 
        plt.show()
                        
        return

#AddBackgroundNoise, AddGaussianNoise, FrequencyMask, TimeMask, TimeStretch
class CustomTensorDataset(data.Dataset):

    def __init__(self, tensors,sample_rate, is_transform=None):
        #print(tensors[0].shape,tensors[1].shape)
        self.tensors = tensors
        self.sample_rate = sample_rate
        self.is_transform = is_transform
        self.augment = Compose([TimeStretch(max_rate=1.75),FrequencyMask(max_frequency_band=0.7),])
        #self.augment = Compose([Gain(),ShuffleChannels(),Shift()])
        
    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]

        if self.is_transform:
            #torch_augs
            #xi = self.augment(x)
            #xj = self.augment(x)

            #normal_augs

            xi = [self.augment(x[i].numpy(),self.sample_rate) for i in range(len(x))]
            xj = [self.augment(x[i].numpy(),self.sample_rate) for i in range(len(x))]

            xi = torch.tensor(np.array(xi))
            xj = torch.tensor(np.array(xj))

            return (xi,xj),y
        else:

            return x, y

    def __len__(self):
        return self.tensors[0].size(0)

def load_runs(args,tr):
    ss = '_frozen' if args['frozen'] else '_unfrozen'
    ss = '' if args['model_type'] == 'sup' else ss
    
    name = '../plots/%s_%s/labs_encoding_%d_encoder_%d%s'%(
        args['data_type'],args['model_type'],args['encoding_size'],args['encoder_type'],ss)
    
    obj = pickle.load(open(name+'.pkl', 'rb'))
    val_accs_dict = obj['val_accs_dict']
    test_accs_dict = obj['test_accs_dict']
    test_losses_dict = obj['test_losses_dict']
    
    cv =args['cv']
    acc =[]
    for j in tr:
        ids = [np.argmax(val_accs_dict[j][i]) for i in range(cv)]
        nums_acc = [test_accs_dict[j][i][ids[i]] for i in range(cv)]
        nums_loss = [test_losses_dict[j][i][ids[i]] for i in range(cv)]
        print('Label (%)',j,'Test Acuracy:', np.mean(nums_acc),max((np.mean(nums_acc) - min(nums_acc)),(max(nums_acc)-np.mean(nums_acc))))
        acc.append(np.mean(nums_acc))
        
    return acc