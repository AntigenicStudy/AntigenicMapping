import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    
    input_path = sys.argv[1]
    test_pred_file = sys.argv[2]
    test_gt_file = sys.argv[3]
    train_pred_file = sys.argv[4]
    train_gt_file = sys.argv[5]
    loss_file = sys.argv[6]
    output_path = sys.argv[7]
    train_img = sys.argv[8]
    test_img = sys.argv[9]

    y_pred_train = np.loadtxt(os.path.join(input_path, train_pred_file))
    y_gt_train = np.loadtxt(os.path.join(input_path, train_gt_file))
    print('Imported result in training!')
    print(y_pred_train.shape)
    print(y_gt_train.shape)

    lossses = np.loadtxt(os.path.join(input_path, loss_file))

    xx = np.arange(round(np.max(y_gt_train)))
    fig = plt.figure(figsize=(5, 5))
    left, bottom, width, height =  0.13,0.125,0.85,0.85
    ax1 = fig.add_axes([left,bottom,width,height])
    y_higher = xx + np.sqrt(2) 
    y_lower = xx - np.sqrt(2) 
    plt.plot(xx,xx, 'b--')
    plt.plot(xx,y_higher, 'c--')
    plt.plot(xx,y_lower, 'c--')

    line2 = plt.scatter(y_gt_train, y_pred_train,  c='red', alpha=0.2)

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    fig.tight_layout()
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.text(0, 20, 'A/H1N1\nTraining', fontsize=22)
    plt.ylabel('Actual antigenic distance', fontsize=20.4)
    plt.xlabel('Predicted antigenic distance', fontsize=20.4)

    xx = np.arange(len(lossses))
    left, bottom, width, height = 0.686,0.22,0.28,0.28
    plt.axes([left,bottom,width,height])
    plt.plot(xx,lossses, 'b--')
    fig.tight_layout()
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.xlabel('Index of epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.savefig(os.path.join(output_path, train_img))

    rmse_train = np.sqrt(np.sum(pow(y_gt_train[:] - y_pred_train[:], 2))/len(y_gt_train))
    mean_dis = np.mean(np.abs(np.abs(y_gt_train[:]) - np.abs(y_pred_train[:])))
    sigma_dis = np.std(np.abs(y_gt_train - y_pred_train))
    print("RMSE in train is " + str(rmse_train))
    print("mean in train is " + str(mean_dis))
    print("sd in train is " + str(sigma_dis))

    print('Maximum of Y is ' + str(np.max(y_gt_train)) + '. Minimum of Y is ' + str(np.min(y_gt_train)))
    print('Maximum of Y is ' + str(np.max(y_gt_train)) + '. Minimum of Y is ' + str(np.min(y_gt_train)))




    y_pred_test = np.loadtxt(os.path.join(input_path, test_pred_file))
    y_gt_test = np.loadtxt(os.path.join(input_path, test_pred_file))

    print('Imported result in test!')
    print(y_pred_test.shape)
    print(y_gt_test.shape)

    xx = np.arange(round(np.max(y_gt_test)))

    fig = plt.figure(figsize=(5,5))
    left, bottom, width, height =  0.13,0.125,0.85,0.85
    ax1 = fig.add_axes([left, bottom, width, height])
    plt.plot(xx,xx, 'b--')
    y_higher = xx + np.sqrt(2) 
    y_lower = xx - np.sqrt(2) 
    plt.plot(xx,y_higher, 'c--')
    plt.plot(xx,y_lower, 'c--')
    line2 = plt.scatter(y_gt_test, y_pred_test, c='red', alpha=0.25)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    fig.tight_layout()
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.text(0, 21, 'A/H1N1\nTest', fontsize=22)
    plt.ylabel('Actual antigenic distance', fontsize=20.4)
    plt.xlabel('Predicted antigenic distance', fontsize=20.4)

    b = list(range(0,round(np.max(y_gt_train)),1))
    left, bottom, width, height = 0.686,0.22,0.28,0.28
    plt.axes([left,bottom,width,height])
    plt.hist(y_pred_test, bins = b, facecolor="red", edgecolor='black', alpha=0.6)
    plt.hist(y_gt_test, bins = b, facecolor="blue", edgecolor='black', alpha=0.6)
    fig.tight_layout()
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.xlabel('Antigenic distance', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.savefig(os.path.join(output_path, test_img))


    rmse_test = np.sqrt(np.sum(pow(y_gt_test[:] - y_pred_test[:], 2))/len(y_pred_test))
    mean_dis = np.mean(np.abs(np.abs(y_gt_test[:]) - np.abs(y_pred_test[:])))
    sigma_dis = np.std(np.abs(y_gt_test - y_pred_test))
    print("RMSE in train is " + str(rmse_test))
    print("mean in train is " + str(mean_dis))
    print("sd in train is " + str(sigma_dis))



if __name__ == '__main__':
    main()