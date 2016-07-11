import numpy as np
import theano
import os

class split_data:
    # Can shuffle return new indices, if so save also, for comparison
    def split(self, filename, step, serieslength):


        # Remove old files
        os.remove('../../data/Joe/pre_proc_lstm.npz')
        os.remove('../../data/Joe/sequential_final_frame.npz')
        os.remove('../../data/Joe/final_frame.npz')
        os.remove('../../data/Joe/edin_shuffled.npz')

        edin = np.load('../../data/Joe/data_edin_locomotion.npz')['clips']
        data = np.load('../../data/Joe/HiddenActivations.npz')['Orig']

        '''
            Shuffle in unison the original data set, the hiddens and the
        '''

        def shuffle_in_unison_inplace(a, b):
            assert len(a) == len(b)
            p = np.random.permutation(len(a))
            return a[p], b[p]

        data, edin = shuffle_in_unison_inplace(data, edin)

        np.savez_compressed('../../data/Joe/edin_shuffled.npz')

        # 30 is the time series length
        data = np.swapaxes(data, 2, 1)
        
        data_std = data.std()
        data_mean = data.mean(axis=2).mean(axis=0)[np.newaxis, :, np.newaxis]

        np.savez_compressed('../../data/Joe/pre_proc_lstm.npz', mean=data_mean, std=data_std)

        data = (data - data_mean) / data_std



        train = data[:310]
        test = data[310:321]

        #X is everything but the final one
        train_x = train[:,:-1]
        # y is the final frame
        train_y = np.squeeze(train[:, -1:])
        test_x = test[:, :-1]
        test_y = np.squeeze(test[:, -1:])

        train_x_dis = train_x
        # Y is every n+1 frame
        train_y_dis = train[:, 1:]
        test_x_dis = test_x
        test_y_dis = test[:, 1:]

        np.savez_compressed('../../data/Joe/sequential_final_frame', train_x=train_x_dis, train_y=train_y_dis, test_x=test_x_dis, test_y=test_y_dis)
        np.savez_compressed('../../data/Joe/final_frame', train_x=train_x, train_y=train_y, test_x=test_x,
                            test_y=test_y)


this = split_data()
this.split('',0,0)
