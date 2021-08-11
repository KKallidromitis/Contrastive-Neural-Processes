import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

def count_parameters(model):
    return sum([p.numel() for p in model.parameters()])

class Conv1DEncoder(nn.Module):
    def __init__(self, input_size,encoding_size,method=None):
        super(Conv1DEncoder, self).__init__()

        self.encoding_size = encoding_size
        self.input_size = input_size
        self.method = method
        
        self.features = nn.Sequential(
            nn.Conv1d(self.input_size, 16, kernel_size=4, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(16, eps=0.001),
	        nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(16, eps=0.001),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(16, eps=0.001),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # nn.Dropout(0.5),

            nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(16, eps=0.001),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # nn.Dropout(0.5),

            nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(16, eps=0.001),
            # nn.Dropout(0.5),
	        nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(16, eps=0.001),
            nn.MaxPool1d(kernel_size=2, stride=2)
            )

        self.fc = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Linear(624, 256),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(256, eps=0.001),
            nn.Linear(256, self.encoding_size)
        )

    def forward(self, x):
        x = self.features(x)
        #print(x.shape)
        if self.method=='NP':
            return x
        x = x.view(x.size(0), -1)
        encoding = self.fc(x)
        return encoding
        
class RnnEncoder(torch.nn.Module):
    def __init__(self,in_channel, encoding_size,device='cpu',hidden_size=64, cell_type='GRU', num_layers=1, dropout=0, bidirectional=True,method=None):
        super(RnnEncoder, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.in_channel = in_channel
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.encoding_size = encoding_size
        self.bidirectional = bidirectional
        self.device = device
        
        self.nn = torch.nn.Sequential(torch.nn.Linear(self.hidden_size*(int(self.bidirectional) + 1), self.encoding_size)).to(self.device)
        
        if cell_type=='GRU':
            self.rnn = torch.nn.GRU(input_size=self.in_channel, hidden_size=self.hidden_size, num_layers=num_layers,
                                    batch_first=False, dropout=dropout, bidirectional=bidirectional).to(self.device)

        elif cell_type=='LSTM':
            self.rnn = torch.nn.LSTM(input_size=self.in_channel, hidden_size=self.hidden_size, num_layers=num_layers,
                                    batch_first=False, dropout=dropout, bidirectional=bidirectional).to(self.device)
        else:
            raise ValueError('Cell type not defined, must be one of the following {GRU, LSTM, RNN}')
            
    def forward(self, x):
        x = x.permute(2,0,1)
        #print(x.shape)
        if self.cell_type=='GRU':
            past = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), x.shape[1], self.hidden_size).to(self.device)
        elif self.cell_type=='LSTM':
            h_0 = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), (x.shape[1]), self.hidden_size).to(self.device)
            c_0 = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), (x.shape[1]), self.hidden_size).to(self.device)
            past = (h_0, c_0)
        
        out, _ = self.rnn(x.to(self.device), past)  # out shape = [seq_len, batch_size, num_directions*hidden_size]

        if self.method == 'NP':
            return out.permute(1,2,0)
    
        encodings = self.nn(out[-1].squeeze(0))
        return encodings

class Resnet1d(nn.Module):

    def __init__(self, input_size,encoding_size,n_feature_maps=32,method=None):
        super(Resnet1d, self).__init__()
        self.method = method
        self.n_feature_maps = n_feature_maps
        self.input_size = input_size
        self.encoding_size = encoding_size
        self.p = 0.5
        #Block 1
        self.conv_x1 = nn.Sequential(
            nn.Conv1d(self.input_size, self.n_feature_maps, kernel_size=8),
            nn.BatchNorm1d(self.n_feature_maps),
            nn.ReLU(inplace=True),
            nn.Dropout(self.p),
        )

        self.conv_y1 = nn.Sequential(
            nn.Conv1d(self.n_feature_maps, self.n_feature_maps, kernel_size=5),
            nn.BatchNorm1d(self.n_feature_maps),
            nn.ReLU(inplace=True),
            nn.Dropout(self.p),
        )
        
        self.conv_z1 = nn.Sequential(
            nn.Conv1d(self.n_feature_maps, self.n_feature_maps, kernel_size=3),
            nn.BatchNorm1d(self.n_feature_maps),
            nn.ReLU(inplace=True),
            nn.Dropout(self.p),
        )
        
        self.shortcut_y1 = nn.Sequential(
            nn.Conv1d(self.input_size, self.n_feature_maps, kernel_size=1),
            nn.BatchNorm1d(self.n_feature_maps),
            nn.Dropout(self.p),
        )
        
        #Block 2,3
        self.conv_x2 = nn.Sequential(
            nn.Conv1d(self.n_feature_maps, self.n_feature_maps*2, kernel_size=8),
            nn.BatchNorm1d(self.n_feature_maps*2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.p),
        )

        self.conv_y2 = nn.Sequential(
            nn.Conv1d(self.n_feature_maps*2, self.n_feature_maps*2, kernel_size=5),
            nn.BatchNorm1d(self.n_feature_maps*2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.p),
        )
        
        self.conv_z2 = nn.Sequential(
            nn.Conv1d(self.n_feature_maps*2, self.n_feature_maps*2, kernel_size=3),
            nn.BatchNorm1d(self.n_feature_maps*2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.p),
        )
        
        self.shortcut_y2 = nn.Sequential(
            nn.Conv1d(self.n_feature_maps, self.n_feature_maps*2, kernel_size=1),
            nn.BatchNorm1d(self.n_feature_maps*2),
            nn.Dropout(self.p),
        )   
        
        #Block 3
        self.conv_x3 = nn.Sequential(
            nn.Conv1d(self.n_feature_maps*2, self.n_feature_maps*2, kernel_size=8),
            nn.BatchNorm1d(self.n_feature_maps*2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.p),
        )
        
        #FC
        self.bn = nn.BatchNorm1d(self.n_feature_maps*2)
        self.av = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(self.n_feature_maps*2, self.encoding_size),
        )
        return
    
    def forward(self, x0):
        #block 1

        x1 = self.conv_x1(F.pad(x0, (1, 6)))
        x1 = self.conv_y1(F.pad(x1, (1, 3)))
        x1 = self.conv_z1(F.pad(x1, (1, 1)))
        
        xs1 = self.shortcut_y1(x0)
        out1 = F.relu(xs1 + x1)
        
        #block 2
        x2 = self.conv_x2(F.pad(out1, (1, 6)))
        x2 = self.conv_y2(F.pad(x2, (1, 3)))
        x2 = self.conv_z2(F.pad(x2, (1, 1)))
        
        xs2 = self.shortcut_y2(out1)
        out2 = F.relu(xs2 + x2)
        
        #block 2
        x2 = self.conv_x2(F.pad(out1, (1, 6)))
        x2 = self.conv_y2(F.pad(x2, (1, 3)))
        x2 = self.conv_z2(F.pad(x2, (1, 1)))
        
        xs2 = self.shortcut_y2(out1)
        out2 = F.relu(xs2 + x2)
        
        #block 3
        x3 = self.conv_x3(F.pad(out2, (1, 6)))
        x3 = self.conv_y2(F.pad(x3, (1, 3)))
        x3 = self.conv_z2(F.pad(x3, (1, 1)))
        
        out3 = F.relu(self.bn(out2) + x3)

        if self.method=='NP':
            return out3
        
        out = self.av(out3)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

def select_encoder(device,encoder_type,input_size,encoding_size,n_classes=None,classifier_type=None,method=None):
    classifier = None
    
    if encoder_type==0:
        encoder = Resnet1d(input_size,encoding_size,method=method).to(device)
    
    elif encoder_type==1:
        encoder = Conv1DEncoder(input_size,encoding_size,method=method).to(device)
            
    elif encoder_type==2:
            encoder = RnnEncoder(input_size, encoding_size,device,method=method).to(device)

    if classifier_type != None:
        classifier = select_classifier(n_classes,encoding_size,classifier_type).to(device)
       
    print('Size of Encoder:', count_parameters(encoder))
    if method=='NP':
        return encoder
    
    return encoder,classifier
    
def select_classifier(n_classes,encoding_size,classifier_type):
    if classifier_type==0:
        classifier = nn.Sequential(
            nn.Linear(encoding_size, n_classes)
        )

    elif classifier_type==1:
        classifier = nn.Sequential(
            nn.Linear(encoding_size, encoding_size),
            nn.ELU(inplace=True),
            nn.Linear(encoding_size, n_classes)
        )

    elif classifier_type==2:
        classifier = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Linear(encoding_size, encoding_size),
            nn.ELU(inplace=True),
            nn.Linear(encoding_size, encoding_size),
            nn.ELU(inplace=True),
            nn.Linear(encoding_size, n_classes)
        )

    return classifier