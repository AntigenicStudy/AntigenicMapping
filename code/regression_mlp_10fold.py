#MLP for mapping embedding distance to antigenic distance

import os
import time
import math
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from numpy import random
from d2l import torch as d2l
import sys

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

loss = nn.MSELoss()

def init_weights(m):
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
            torch.nn.init.xavier_uniform_(m.weight)

def MLPNet(neuron_1, neuron_2, neuron_3, neuron_4):
    print('enter MLP net')
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(565, neuron_1),
                        nn.ReLU(),
                        nn.Linear(neuron_1, neuron_2),
                        nn.ReLU(),
                        nn.Linear(neuron_2, neuron_3),
                        nn.ReLU(),
                        nn.Linear(neuron_3, neuron_4),
                        nn.ReLU(),
                        nn.Linear(neuron_4, 1))
    net.apply(init_weights)
    return net

def log_rmse(net, features, labels):
        clipped_preds = torch.clamp(net(features), 1, float('inf'))
        rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
        return rmse.item()

def rmse(y_gt,y_pred):
    rmse_test = np.sqrt(np.sum(pow(y_gt[:] - y_pred[:], 2))/len(y_pred))

def train(net, train_features, train_labels, test_features, test_labels,
        num_epochs, learning_rate, weight_decay, batch_size, cuda_id, kfold_str, opt_nm, gt, model_path):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    lowest_rmse = 1e9
    if opt_nm == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(),
                                lr = learning_rate,
                                weight_decay = weight_decay)
    if opt_nm == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(),
                                lr = learning_rate,
                                weight_decay = weight_decay)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
            
        #scheduler.step()
        temp_rmse = log_rmse(net, train_features, train_labels)
        if np.any(np.isnan(temp_rmse)):
            print('rmse is null, training break')
            break
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
        end_time = time.time()
        
        y_pred = net(test_features).cpu().detach()
        y_pred_np = np.array(y_pred)
        temp_sum = 0
        rmse_redo = 0
        for i in range(len(y_pred_np)):
            temp_sum = temp_sum + pow(gt[i] - y_pred_np[i],2)
        rmse_redo = math.sqrt(temp_sum / len(gt))
        
        print(cuda_id + ' ' + kfold_str + ' ' + str(round(lowest_rmse,3)) + ' ' + str(epoch) + '/' + str(num_epochs) + ' log rmse is ' + str(round(train_ls[-1],4)) + ' running time is ' + str(round(end_time - start_time,2)) + ' secs')
        
        if rmse_redo < lowest_rmse:
            lowest_rmse = rmse_redo
            torch.save(net.state_dict(), model_path + 'best-' + kfold_str + '-' + str(epoch) + '.mdl')   
            print('save model, epoch is: ' + str(epoch) + ' lowest rmse is: ' + str(rmse_redo)) 
            
    return train_ls, test_ls

def train_and_pred(train_features, test_features, train_labels, test_data,
                num_epochs, lr, weight_decay, batch_size, cuda_id, kfold_str,neuron_1, neuron_2, neuron_3, device, opt_nm, gt,model_path,image_path):
    net = MLPNet(neuron_1, neuron_2, neuron_3, neuron_4).to(device)
    start = time.time()
    train_ls, test_ls = train(net, train_features, train_labels, test_features, test_data,
                        num_epochs, lr, weight_decay, batch_size,cuda_id,kfold_str, opt_nm,gt,model_path)
    end = time.time()
    plt.figure(figsize=(5,5))
    plt.plot(train_ls,'r--')
    plt.plot(test_ls,'b--')
    plt.title('Loss curve')
    plt.ylabel('log rmse')
    plt.xlabel('epoch')
    plt.tight_layout()
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.savefig(image_path + kfold_str + '-best.png')
            
    print(f'train log rmse: {float(train_ls[-1]):f}')
    print('training time is ' + str(end - start))


if __name__ == '__main__':
    cuda_id = '0'

    lowest_rmse = 1e8
    num_epochs = 700
    kfold_str_list = ['2','3','4','5','6','7','8','9']
    kfold_str_list_outer = ['2']
    kfold_str_list_inner = ['0','1','2','3','4','5','6','7','8','9']
    kfold_str_outer = '0'
    kfold_str_inner = '0'
    
    lr = 9e-5
    batch_sz = 256
    lr_list = [2e-4,3e-4]
    batch_sz_list = [32,64,128,256,512,1024]
    wd_list = [0,1e-6,1e-5,1e-3,0.01]
    
    wt_dc = 0
    
    neuron_1 = 4096
    neuron_2 = 2048
    neuron_3 = 1024
    neuron_4 = 48
    
    path = ''
    model_path = ''
    image_path = '' 
    
    for lr_str in lr_list:
        lr = lr_str
        for bs in batch_sz_list:
            batch_sz = bs
            for wd in wd_list:
                wt_dc = wd
                for fold_outer in kfold_str_list_outer:
                    kfold_str_outer = fold_outer
                    path = '4layer-lowestepoch/' + kfold_str_outer + '/' + str(neuron_1) + '-' + str(neuron_2) + '_' + str(neuron_3) + '_' + str(neuron_4) + '_' + opt_nm + '_' + str(lr) + '_' + str(batch_sz)  + '_' +  str(wt_dc)  
                    os.makedirs(path)
                    os.makedirs(path + '/models')
                    os.makedirs(path + '/images')
                    os.makedirs(path + '/results')
                    os.makedirs(path + '/plot')
                    for fold_inner in kfold_str_list_inner:
                        kfold_str_inner = fold_inner 
                        model_path = path + '/models/'
                        image_path = path + '/images/' 
                        main()  

