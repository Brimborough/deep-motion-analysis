import numpy as np
#TODO: hopefully delete.

def find():
    data = np.load('../data/Joe/split_data.npz')
    train_x = data['train_x']

    pre = np.load('../data/Joe/pre_proc_lstm.npz')

    train_x = (train_x*pre['std'][:,-1])
    train_x = train_x +pre['mean'][:,-1]

    dat = np.load('../data/Joe/HiddenActivations.npz')['Orig']
    dat = np.swapaxes(dat,1,2)
    dat = dat[:,:-1]

    print train_x[0].shape
    print dat[0].shape

    print

    for i in range(321):
        if (train_x[0] - dat[i]).sum() < .0001:
            print i

find()