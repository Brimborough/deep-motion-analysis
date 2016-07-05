import numpy as np
import theano
import theano.tensor as T
import imdb

import sys
sys.path.append('../representation_learning/nn')
from LSTM import LSTM

from Network import Network
from AdamTrainer import AdamTrainer


rng = np.random.RandomState(23455)

shared = lambda d: theano.shared(d)


#Make them into numpy arrays - then we can access shape variable.
train, valid, test = imdb.load_data()

train_set_x, train_set_y = map(shared, train)
valid_set_x, valid_set_y = map(shared, valid)
test_set_x, test_set_y   = map(shared, test)

batchsize = 16

"""
    Now prepare the data:
        - Prepared in the iteration of training, therefore, create masks before hand as a big matrix
            put into a shared variable or is there a way to alter the adam trainer?
        - Also need to implemented the get mini-batches outside of adam, I think adam deals with this,
            check.

"""
# Test the outputs from the preperation of the data ensuring they are in the correct order.

network = Network(
	LSTM()
)

trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=5, alpha=0.00001, cost='cross_entropy')
trainer.train(network=network, train_input=train_set_x, train_output=train_set_y,
                               valid_input=valid_set_x, valid_output=valid_set_y,
                               test_input=test_set_x, test_output=test_set_y, filename=None)

