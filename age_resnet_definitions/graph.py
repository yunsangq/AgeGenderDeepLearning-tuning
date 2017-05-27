import matplotlib.pyplot as plt
import numpy as np

train_iter = []
val_iter = []
total_train_loss = []
total_val_acc = []
total_val_loss = []
for fold_number in 18, 34, 50:
    val_log = './'+str(fold_number)+'-model_fold_0/loss_history.log.test'
    train_log = './'+str(fold_number)+'-model_fold_0/loss_history.log.train'

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

    train_iter.append(list(train_log_info[0]))
    val_iter.append(list(val_log_info[0]))

    total_train_loss.append(list(train_log_info[2]))
    total_val_acc.append(list(val_log_info[2]))
    total_val_loss.append(list(val_log_info[3]))


def err_disp(train_iter, valid_iter, train, valid):
    fig = plt.figure(facecolor='white')
    fig.canvas.set_window_title('Cost per Iters')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(train_iter[1:], train[1:], color='#1F77B4', label='Training')
    ax.plot(valid_iter[1:], valid[1:], color='#b41f1f', label='Validation')
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


def total_err_disp(valid_iter, valid):
    fig = plt.figure(facecolor='white')
    fig.canvas.set_window_title('Cost per Iters')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(valid_iter[0][1:], valid[0][1:], color='#1F77B4', label='18-resnet valid')
    ax.plot(valid_iter[1][1:], valid[1][1:], color='#b41f1f', label='34-resnet valid')
    ax.plot(valid_iter[2][1:], valid[2][1:], color='#000000', label='50-resnet valid')
    ax.set_xlabel('iter')
    ax.set_ylabel('Cost')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('./cost.png')
    #plt.show()


def total_acc_disp(valid_iter, valid):
    fig = plt.figure(facecolor='white')
    fig.canvas.set_window_title('Accuracy per Iters')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(valid_iter[0], valid[0], color='#1F77B4', label='18-resnet valid')
    ax.plot(valid_iter[1], valid[1], color='#b41f1f', label='34-resnet valid')
    ax.plot(valid_iter[2], valid[2], color='#000000', label='50-resnet valid')
    ax.set_xlabel('iter')
    ax.set_ylabel('Accuracy')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('./acc.png')
    #plt.show()


total_err_disp(val_iter, total_val_loss)
#total_acc_disp(val_iter, total_val_acc)
#err_disp(train_iter[2], val_iter[2], total_train_loss[2], total_val_loss[2])
# acc_disp(val_iter[1], total_val_acc[1])
