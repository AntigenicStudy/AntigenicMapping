import os
import time
import torch
import math
import sys
import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.utils import data as Data
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from numpy import random


init_channel = 64

class Residual(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
            
        super(Residual, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

        self.b1 = torch.nn.BatchNorm2d(out_channels)
        self.b2 = torch.nn.BatchNorm2d(out_channels)
 
    def forward(self, X):
        Y = F.relu(self.b1(self.conv1(X)))
        Y = self.b2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        return F.relu(Y + X)


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
        
        if first_block:
            assert in_channels == out_channels
        block = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                block.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=1))
            else:
                block.append(Residual(out_channels, out_channels))
        return torch.nn.Sequential(*block)


def ResNet():
    b1 = nn.Sequential(
         nn.Conv2d(1, init_channel, kernel_size=2, padding=1, stride=1), 
         nn.BatchNorm2d(init_channel), 
         nn.ReLU(),
         nn.MaxPool2d(kernel_size=2)
    )
    
    b2 = nn.Sequential(*resnet_block(init_channel, init_channel, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(init_channel, init_channel * 2, 2))
    b4 = nn.Sequential(*resnet_block(init_channel * 2, init_channel * 4, 2))
    b5 = nn.Sequential(*resnet_block(init_channel * 4, init_channel * 8, 2))

    net = nn.Sequential(b1, b2, b3, b4, b5)
          
    net.add_module("global_avg_pool", GlobalAvgPool2d())
    net.add_module("flatten", FlattenLayer())
    net.add_module("fc", torch.nn.Linear(init_channel * 8, 1))
    
    return net


class AnalysisDataset(Dataset):
    def __init__(self, in_data, column_X, column_Y):
        self.data = in_data
        self.column_X = column_X
        self.column_Y = column_Y
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        X = self.data[self.column_X]
        y = self.data[self.column_Y]
        tensor_X = torch.from_numpy(np.array(X.iloc[index])).float()
        tensor_y = torch.from_numpy(np.array(y.iloc[index])).float()
        return tensor_X.view(1, tensor_X.shape[0], 1), tensor_y


class PredictDataset(Dataset):
    def __init__(self, in_data, column_X):
        self.data = in_data
        self.column_X = column_X
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        X = self.data[self.column_X]
        tensor_X = torch.from_numpy(np.array(X.iloc[index])).float()
        return tensor_X.view(1, tensor_X.shape[0], 1)


def loadData(batch_size, X, y):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=list(np.linspace(0, X.shape[1] -1, X.shape[1])))
 
    if not isinstance(y, pd.DataFrame):
        y = pd.DataFrame(y, columns=list(np.linspace(X.shape[1], X.shape[1] + y.shape[1] - 1, y.shape[1])))
    column_X = X.columns
    column_y = y.columns
    temp_df = X.join(y)

    train_set, test_set = train_test_split(temp_df, test_size=0.2, random_state=10, shuffle=True)
 
    torch_trainset = AnalysisDataset(train_set, column_X, column_y)
    torch_testset = AnalysisDataset(test_set, column_X, column_y)
    
    works_num = 4
 
    train_batch = DataLoader(torch_trainset, batch_size=batch_size, shuffle=True, num_workers=works_num)
    test_batch = DataLoader(torch_testset, batch_size=batch_size, shuffle=False, num_workers=works_num)
 
    return train_batch, test_batch


class GlobalAvgPool2d(torch.nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
       return F.avg_pool2d(x, kernel_size=x.size()[2:])


class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)

lossses = []

def train(model, train_batch, test_batch, batch_size, optimizer, device, num_epochs):
    model = model.to(device)
    print("run in " , device)
 
    loss = torch.nn.MSELoss()
 
    for epoch in range(num_epochs):
        train_loss_sum, train_rmse_sum, n, batch_count = 0.0, 0.0, 0, 0
        start = time.time()
 
        for X, y in train_batch:
            X = X.to(device)
            y = y.to(device)
 
            y_pre = model(X)
 
            l = loss(y_pre, y)
 
            optimizer.zero_grad()
 
            l.backward()
            optimizer.step()
 
            train_loss_sum += l.cuda().item()
            train_rmse_sum += torch.sqrt(((y_pre-y)**2).sum()).cuda().item()
            n += y.shape[0]
            batch_count += 1
 
        test_rmse = evaluate_rmse(test_batch, model,device)
 
        print("epoch:%d / %d, loss:%.4f, train_rmse:%.3f, test_rmse %.3f, cost: %.1f sec" %
              (epoch + 1, num_epochs, train_loss_sum / batch_count, train_rmse_sum / n, test_rmse, time.time() - start))
        
        lossses.append(train_loss_sum / batch_count)


def evaluate_rmse(data_batch, model, device):
    device = torch.device('cuda')
    
    rmse_sum, n = 0, 0
 
    with torch.no_grad():
        for X, y in data_batch:
            if isinstance(model, torch.nn.Module):
                model.eval()
                #pdb.set_trace();
                
                rmse_sum += torch.sqrt(((model(X.to(device)) - y.to(device)) ** 2).sum()).cuda().item()               
                model.train()
            else:
                if ('is_training' in model.__code__.co_varnames):
                    rmse_sum += torch.sqrt(((model(X.to(device), is_training=False) - y) ** 2).sum()).cuda().item()
                else:
                    rmse_sum += torch.sqrt(((model(X.to(device)) - y) ** 2).sum()).cuda().item()
            n += y.shape[0]
    return rmse_sum / n


def validation(model, test_batch, device):
    device = torch.device('cuda')
    model = model.to(device)
    predX, predy = iter(test_batch).next()
 
    rmse_sum, n = 0, 0
 
    with torch.no_grad():
        if isinstance(model, torch.nn.Module):
            rmse_sum += torch.sqrt(((model(predX.to(device)) - predy.to(device)) ** 2).sum()).cuda().item()
        else:
            if ('is_training' in model.__code__.co_varnames):
                rmse_sum += torch.sqrt(((model(predX.to(device), is_training=False) - predy) ** 2).sum()).cuda().item()
            else:
                rmse_sum += torch.sqrt(((model(predX.to(device)) - predy) ** 2).sum()).cuda().item()
        n += predy.shape[0]
        
    return rmse_sum / n


def predict(model, df_X, batch_size, device):
    
    device = torch.device('cuda') 
    model = model.to(device)
    
    if not isinstance(df_X, pd.DataFrame):
        df_X = pd.DataFrame(df_X, columns=list(np.linspace(0, df_X.shape[1] -1, df_X.shape[1])))
 
    predict_dataset = PredictDataset(df_X, df_X.columns)
 
    works_num = 4
 
    predict_batch = Data.DataLoader(predict_dataset, batch_size=batch_size, shuffle=False, num_workers=works_num)
 
    estimated_Y = torch.tensor([])
    estimated_Y.to(device)
    for X in predict_batch:
        
        X = X.to(device)  
        temp_Y = model(X)
        
        estimated_Y = torch.cat([estimated_Y.to(device), temp_Y.to(device)], dim=0).to(device)
        estimated_Y = estimated_Y.detach()
       
    return estimated_Y


def main():

    distance_path = sys.argv[1]
    dis_file = sys.argv[2]
    embedding_path = sys.argv[3]
    embedding_file = sys.argv[4]
    output_path = sys.argv[5]
    loss_file = sys.argv[6]
    model_file = sys.argv[7]
    test_pred_file = sys.argv[8]
    test_gt_file = sys.argv[9]
    train_pred_file = sys.argv[10]
    train_gt_file = sys.argv[11]

    ''' Import and show X and Y values '''
    dis = np.loadtxt(os.path.join(distance_path, dis_file))

    dis_flat = dis.flatten()[:]
    print('Imported antigenic distance!')
    print('Shape of Y is ' + str(dis_flat.shape))
    print('Maximum of Y is ' + str(np.max(dis_flat)) + '. Minimum of Y is ' + str(np.min(dis_flat)) + '\n') 

    data_set_input = np.loadtxt(os.path.join(embedding_path, embedding_file))
    print(data_set_input.shape)
    
    # remove duplicate distance value
    rows = []
    idx = np.arange(dis_flat.shape[0])
    for i in range(2):
        for j in range(dis.shape[0]):
            for k in idx[j * dis.shape[0] : j * dis.shape[0] + j + 1]:
                rows.append(k)
            
    data_set_input = np.delete(data_set_input, rows, axis=0)

    data_set_output = np.array(dis_flat)[:]
    data_set_output = np.delete(data_set_output, rows, axis=0)
    
    print('Imported embedding distance!')
    print('Shape of X is '+str(data_set_input.shape) + ' Shape of Y is ' + str(data_set_output.shape) + '\n')
    

    ''' Start train'''
    indices = np.random.permutation(data_set_input.shape[0])
    training_idx, test_idx = indices[:int(0.8 * len(indices))], indices[int(0.8 * len(indices)):]

    x = []
    y = []
    for i in range(len(training_idx)):
        x.append(data_set_input[training_idx[i]])
        y.append(data_set_output[training_idx[i]])
    x = np.array(x)
    y = np.array(y)
    y = y[:,np.newaxis]
    y_train_gt = y
    print('Start training')
    print('Shape of X in training is '+str(x.shape) + ' Shape of Y in training is ' + str(y.shape))
    print('Maximum of Y in training is ' + str(np.max(y)) + '. Minimum of Y in training is ' + str(np.min(y)) + '\n') 
    max_y = np.max(y)

    batch_size  = 64
    train_batch, test_batch = loadData(batch_size, x, y)
    
    lr, num_epochs = 3e-5, 5
    device = torch.device('cuda')
    model = ResNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    train(model, train_batch, test_batch, batch_size, optimizer, device, num_epochs)
    y_train = predict(model, x, batch_size,device)

    y_train[y_train > max_y] = max_y
    y_train = y_train.cpu().numpy()
    y_train_txt = y_train
    xx = np.arange(round(max_y))

    valid_rmse = validation(model, test_batch,device)
    print("validation rmse:" + str(valid_rmse) + '\n')
    
    f1 = open(os.path.join(output_path, loss_file),'w')
    np.savetxt(f1, lossses, fmt='%f', delimiter=' ', newline=' ')
    f1.close()



    ''' Test the model and save predicted value'''  
    x = []
    y_test = []
    for i in range(len(test_idx)):
        x.append(data_set_input[test_idx[i]])
        y_test.append(data_set_output[test_idx[i]])

    x = np.array(x)
    y_test = np.array(y_test)
    y_test_txt = y_test
    y_test = y_test[:,np.newaxis]

    print('Shape of X in test is '+str(x.shape) + ' Shape of Y in test is ' + str(y_test.shape))
    print('Finish predict!')

    torch.save(model.state_dict(), os.path.join(output_path, model_file))

    y = predict(model, x, batch_size,device)
    y = y.cpu().numpy()
    
    y_test = np.array(y_test)
    f1 = open(os.path.join(output_path, test_pred_file),'w')
    np.savetxt(f1, y_test_txt, fmt='%f', delimiter=' ', newline=' ')
    f1.close()

    f1 = open(os.path.join(output_path, test_gt_file),'w')
    np.savetxt(f1, y, fmt='%f', delimiter=' ', newline=' ')
    f1.close()

    f1 = open(os.path.join(output_path, train_pred_file),'w')
    np.savetxt(f1, y_train_txt, fmt='%f', delimiter=' ', newline=' ')
    f1.close()

    f1 = open(os.path.join(output_path, train_gt_file),'w')
    np.savetxt(f1, y_train_gt, fmt='%f', delimiter=' ', newline=' ')
    f1.close()
    print('Saved results!')

if __name__ == '__main__':
    main()
