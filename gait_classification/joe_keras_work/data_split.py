import numpy as np
import theano


class split_data:
    # Can shuffle return new indices, if so save also, for comparison
    def split(self, filename, step, serieslength):
        data = np.load('../data/Joe/HiddenActivations.npz')['Orig']

        # 30 is the time series length
        data = np.swapaxes(data, 2, 1)

        data_std = np.array([[[data.std()]]]).repeat(data.shape[1], axis=1)
        data_mean = data.mean(axis=2).mean(axis=0)[np.newaxis, :, np.newaxis]

        np.savez_compressed('../data/Joe/pre_proc_lstm.npz', mean=data_mean, std=data_std)

        data = (data - data_mean) / data_std

        print data.std()
        print data.mean()

        np.random.shuffle(data)

        train = data[:310]
        test = data[310:321]

        #X is everything but the final one
        train_x = train[:,:-1]
        train_x_dis = train_x
        print train_x_dis.shape

        # y is the final frame
        train_y = np.squeeze(train[:, -1:])
        # Y is every n+1 frame
        train_y_dis = train[:, 1:]
        print train_y_dis.shape

        test_x = test[:,:-1]
        test_x_dis = test_x
        test_y = np.squeeze(test[:,-1:])
        test_y_dis = test[:, 1:]
        print test_y_dis.shape
        '''
        for i in range(0, 255,step):
            np.concatenate(train_x, train[:, i:i+serieslength])
            np.concatenate(train_y, np.squeeze(train[:, i+serieslength:i+serieslength+1]))
            #Do the same for the tests
            np.concatenate(test_x,test[:, i:i + serieslength])
            np.concatenate(test_y,np.squeeze(test[:, i+serieslength:i + serieslength + 1]))
        '''
        np.savez_compressed('../data/Joe/sequential_final_frame', train_x=train_x_dis, train_y=train_y_dis, test_x=test_x_dis, test_y=test_y_dis)
        np.savez_compressed('../data/Joe/final_frame', train_x=train_x, train_y=train_y, test_x=test_x,
                            test_y=test_y)


this = split_data()
this.split('',0,0)
