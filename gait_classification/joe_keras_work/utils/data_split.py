import numpy as np
import theano
import os

class split_data:
    # Can shuffle return new indices, if so save also, for comparison
    def split(self, filename, step, serieslength):


        # Remove old files
        if(os.path.isfile('../../data/pre_proc_lstm_edin_hdm05.npz')):
            os.remove('../../data/pre_proc_lstm_edin_hdm05.npz')
        if (os.path.isfile('../../data/sequential_final_frame_edin_hdm05.npz')):
            os.remove('../../data/sequential_final_frame_edin_hdm05.npz')
        if (os.path.isfile('../../data/final_frame_edin_hdm05.npz')):
            os.remove('../../data/final_frame_edin_hdm05.npz')
        if (os.path.isfile('../../data/edin_hdm05_shuffled.npz')):
            os.remove('../../data/edin_hdm05_shuffled.npz')

        edin = np.load('../../data/edin_hdm05_locomotion.npz')['clips']
        data = np.load('../../data/HiddenActivations_Hdm05_edin.npz')['Orig']

        '''
            Shuffle in unison the original data set, the hiddens and the
        '''

        def shuffle_in_unison_inplace(a, b):
            assert len(a) == len(b)
            p = np.random.permutation(len(a))
            return a[p], b[p]


        data, edin = shuffle_in_unison_inplace(data, edin)

        np.savez_compressed('../../data/edin_hdm05_shuffled.npz', clips=edin)

        # 30 is the time series length
        data = np.swapaxes(data, 2, 1)
        
        data_std = data.std()
        data_mean = data.mean(axis=2).mean(axis=0)[np.newaxis, :, np.newaxis]

        np.savez_compressed('../../datapre_proc_lstm_edin_hdm05.npz', mean=data_mean, std=data_std)

        data = (data - data_mean) / data_std



        train = data[:1000]
        test = data[1000:]

        #X is everything but the final one
        train_x = train[:,:-1]
        # y is the final fram
        train_y = np.squeeze(train[:, -1:])
        test_x = test[:, :-1]
        test_y = np.squeeze(test[:, -1:])

        train_x_dis = train_x
        # Y is every n+1 frame
        train_y_dis = train[:, 1:]
        test_x_dis = test_x
        test_y_dis = test[:, 1:]

        np.savez_compressed('../../data/sequential_final_frame_edin_hdm05', train_x=train_x_dis, train_y=train_y_dis, test_x=test_x_dis, test_y=test_y_dis)
        np.savez_compressed('../../data/final_frame_edin_hdm05', train_x=train_x, train_y=train_y, test_x=test_x,
                            test_y=test_y)


this = split_data()
this.split('',0,0)
