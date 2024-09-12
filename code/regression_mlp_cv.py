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





def main():

    device = torch.device('cuda:' + cuda_id)
    
    ''' Import and show X and Y values '''
    dis = np.loadtxt('antigenicDis/fold_' + kfold_str_outer + '/' + 'h3-dis-train-' + '-' + kfold_str_inner + '.csv', delimiter=',')

    dis_show = dis.flatten()[:]
    gt = np.array(dis_show)[:]
    
    print("size of gt is " + str(gt.shape))
    print('Shape of Y is ' + str(dis_show.shape))
    print('Maximum of Y is ' + str(np.max(dis_show)) + '. Minimum of Y is ' + str(np.min(dis_show))) 

    data_set_input = np.genfromtxt('semanticDis/10_fold_' + kfold_str_outer + '/h3-semanticDis-train-' + kfold_str_inner + '.csv', delimiter=',', usecols=np.arange(0,565)) #[:,0:560]#np.loadtxt('matrix/h3_pairs-nowcast-b.csv', delimiter=',')
    print(data_set_input.shape)
    
    virus_count = int(math.sqrt(data_set_input.shape[0]))

    print('check nan')
    print(np.any(np.isnan(data_set_input)))
    print(np.argwhere(np.isnan(data_set_input)))
    
    #delete duplicate elements in the matrix
    rows = []
    idx = np.arange(virus_count * virus_count)
    for i in range(2):
        for j in range(virus_count):
            for k in idx[j * virus_count : j * virus_count + j + 1]:
                rows.append(k)
            
    data_set_input = np.delete(data_set_input, rows, axis=0)
    antiDis_max = np.max(dis_show)
    data_set_input = data_set_input * antiDis_max
    data_set_output = np.array(dis_show)[:]
    data_set_output = np.delete(data_set_output, rows, axis=0)
    
    print('Shape of X is '+str(data_set_input.shape) + ' Shape of Y is ' + str(data_set_output.shape))
    
    indices = np.random.permutation(data_set_input.shape[0])
    training_idx, test_idx = indices[:int(1 * len(indices))], indices[int(1 * len(indices)):]

    x = []
    y = []
    for i in range(len(training_idx)):
        x.append(data_set_input[training_idx[i]])
        y.append(data_set_output[training_idx[i]])

    train_features = np.array(x)
    train_lables = np.array(y)
    train_lables = train_lables[:,np.newaxis]
    train_features = torch.tensor(train_features, dtype=torch.float32).to(device)
    train_lables = torch.tensor(train_lables,dtype=torch.float32).to(device)
    
    print('Shape of train X is '+str(train_features.shape) + ' Shape of train Y is ' + str(train_lables.shape))
    print('Maximum of train Y is ' + str(torch.max(train_lables)) + '. Minimum of train Y is ' + str(torch.min(train_lables))) 
    
    dis = np.loadtxt('antigenicDis/fold_' + kfold_str_outer + '/' + 'h3-dis-test-' + '-' + kfold_str_inner + '.csv', delimiter=',')
    dis_show = dis.flatten()[:]
    gt = np.array(dis_show)[:]
    
    print('Shape of Y is ' + str(dis_show.shape))
    print('Maximum of Y is ' + str(np.max(dis_show)) + '. Minimum of Y is ' + str(np.min(dis_show))) 
    data_set_input_test = np.genfromtxt('semanticDis/10_fold_' + kfold_str_outer + '/h3-semanticDis-test-' + kfold_str_inner + '.csv', delimiter=',', usecols=np.arange(0,565)) #[:,0:560]#np.loadtxt('matrix/h3_pairs-nowcast-b.csv', delimiter=',')
    print(data_set_input_test.shape)
    
    virus_count = int(math.sqrt(data_set_input_test.shape[0]))
    print('check nan')
    print(np.any(np.isnan(data_set_input_test)))
    print(np.argwhere(np.isnan(data_set_input_test)))
    rows = []
    idx = np.arange(virus_count * virus_count)
    for i in range(2):
        for j in range(virus_count):
            for k in idx[j * virus_count : j * virus_count + j + 1]:
                rows.append(k)
            
    data_set_input_test = np.delete(data_set_input_test, rows, axis=0)
    data_set_input_test = data_set_input_test * antiDis_max
    data_set_output_test = np.array(dis_show)[:]
    data_set_output_test = np.delete(data_set_output_test, rows, axis=0)
    gt = np.delete(gt, rows, axis=0)
    print('Shape of X is '+str(data_set_input_test.shape) + ' Shape of Y is ' + str(data_set_output_test.shape))
    
    x = []
    y = []
    for i in range(len(data_set_output_test)):
        x.append(data_set_input_test[i])
        y.append(data_set_output_test[i])

    test_features = np.array(x)
    test_lables = np.array(y)
    test_lables = test_lables[:,np.newaxis]
    test_features = torch.tensor(test_features, dtype=torch.float32).to(device)
    test_lables = torch.tensor(test_lables,dtype=torch.float32).to(device)
    
    print('Shape of test X is '+str(test_features.shape) + ' Shape of test Y is ' + str(test_lables.shape))
    print('Maximum of test Y is ' + str(torch.max(test_lables)) + '. Minimum of test Y is ' + str(torch.min(test_lables))) 
    
    num_epochs, weight_decay, batch_size = num_epochs, wt_dc, batch_sz
    train_and_pred(train_features, test_features, train_lables, test_lables,
               num_epochs, lr, weight_decay, batch_size, cuda_id, kfold_str_inner, neuron_1, neuron_2, neuron_3, device, opt_nm, gt, model_path, image_path) 





if __name__ == '__main__':
    
    cuda_id = '0'

    lowest_rmse = 1e8
    num_epochs = 1000
    kfold_str_outer = '0'
    kfold_str_inner = '0'
    kfold_str_list_outer = ['0']
    kfold_str_list_inner = ['0','1','2','3','4','5','6','7','8','9']
    
    lr = 0
    lr_list = [2e-4,3e-4]
    
    batch_sz = 0
    batch_sz_list = [32,64,128,256,512,1024]
    
    wt_dc = 0
    wd_list = [0,0.01,0.001,0.0001,0.00001,0.000001]
    
    neuron_1 = 2048
    neuron_2 = 1024
    neuron_3 = 512
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
                    path = '4layer-outcome/' + kfold_str_outer + '/' + str(neuron_1) + '-' + str(neuron_2) + '_' + str(neuron_3) + '_' + str(neuron_4) + '_' + str(lr) + '_' + str(batch_sz)  + '_' +  str(wt_dc)  
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
