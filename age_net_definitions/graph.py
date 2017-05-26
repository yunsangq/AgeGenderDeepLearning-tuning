import matplotlib.pyplot as plt
import numpy as np

train_iter = []
val_iter = []
total_train_loss = []
total_val_acc = []
total_val_loss = []
for fold_number in range(5):
    val_log = './model_fold_'+str(fold_number)+'/loss_history.log.test'
    train_log = './model_fold_'+str(fold_number)+'/loss_history.log.train'

    val_log_info = []
    train_log_info = []

    with open(val_log, 'r') as txt:
        idx = 0
        for line in txt:
            if idx != 0:
                val_log_info.append(line.split())
            idx += 1

    with open(train_log, 'r') as txt:
        idx = 0
        for line in txt:
            if idx != 0:
                train_log_info.append(line.split())
            idx += 1

    train_log_info = zip(*train_log_info)
    val_log_info = zip(*val_log_info)

    if fold_number == 0:
        train_iter = list(train_log_info[0])
        val_iter = list(val_log_info[0])

    total_train_loss.append(list(train_log_info[2]))
    total_val_acc.append(list(val_log_info[2]))
    total_val_loss.append(list(val_log_info[3]))


def err_disp(train_iter, valid_iter, train, valid):
    fig = plt.figure(facecolor='white')
    fig.canvas.set_window_title('Cost per Iters')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(train_iter, train, color='#1F77B4', label='Training')
    ax.plot(valid_iter, valid, color='#b41f1f', label='Validation')
    ax.set_xlabel('iter')
    ax.set_ylabel('Cost')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('./cost.png')
    #plt.show()


def acc_disp(iter, valid):
    fig = plt.figure(facecolor='white')
    fig.canvas.set_window_title('Accuracy per Iters')
    ax = fig.add_subplot(1, 1, 1)
    # ax.plot(iter, train, color='#1F77B4', label='Training')
    ax.plot(iter, valid, color='#b41f1f', label='Validation')
    ax.set_xlabel('iter')
    ax.set_ylabel('Accuracy')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('./acc.png')
    #plt.show()


total_train_loss = np.sum(np.asarray(total_train_loss).astype(np.float), axis=0)/5.
total_val_loss = np.sum(np.asarray(total_val_loss).astype(np.float), axis=0)/5.
total_val_acc = np.sum(np.asarray(total_val_acc).astype(np.float), axis=0)/5.

err_disp(train_iter, val_iter, total_train_loss, total_val_loss)
acc_disp(val_iter, total_val_acc)
