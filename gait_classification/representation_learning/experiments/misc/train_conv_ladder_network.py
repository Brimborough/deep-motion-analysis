import numpy as np
import sys
import theano
import theano.tensor as T

sys.path.append('../..')

from nn.ActivationLayer import ActivationLayer
from nn.AdamTrainer import *
from nn.LadderAdamTrainer import LadderAdamTrainer
from nn.Conv2DLayer import Conv2DLayer
from nn.ConvLadderNetwork import ConvLadderNetwork
from nn.HiddenLayer import HiddenLayer
from nn.Pool2DLayer import Pool2DLayer
from nn.ReshapeLayer import ReshapeLayer

from tools.utils import load_mnist, remove_labels

rng = np.random.RandomState(23455)

datasets = load_mnist(rng)

shared = lambda d: theano.shared(d, borrow=True)
train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y   = datasets[2]

# Create a semi-supervised task by removing all but a hundred labels in the
# training set
train_set_y = remove_labels(rng, train_set_y, train_set_y.shape[0] - 100)

train_set_x = train_set_x.reshape((50000, 1, 28, 28))
valid_set_x = valid_set_x.reshape((10000, 1, 28, 28))
test_set_x  = test_set_x.reshape((10000, 1, 28, 28))

train_set_x, train_set_y = map(shared, [train_set_x, train_set_y])
valid_set_x, valid_set_y = map(shared, [valid_set_x, valid_set_y])
test_set_x, test_set_y   = map(shared, [test_set_x, test_set_y])

l_train_set_x, u_train_set_x = split_data(train_set_x, train_set_y)
l_train_set_y, u_train_set_y = split_data(train_set_y, train_set_y)

batchsize = 100
sigma = 0.3

network = ConvLadderNetwork(
        encoding_layers =
            # Where 2*batchsize is necessary as we pass a minibatch of
            # labeled and unlabeled data simultaneously
            [Conv2DLayer(rng, (4, 1, 5, 5), (2*batchsize, 1, 28, 28)),
            ActivationLayer(rng, f='ReLU'),
            Pool2DLayer(rng, (2*batchsize, 4, 28, 28)),

            ReshapeLayer(rng, shape=(2*batchsize, 4*14*14)),
            HiddenLayer(rng, (4*14*14, 10)),
            ActivationLayer(rng, f='softmax')],

        decoding_layers =
            [HiddenLayer(rng, (4*14*14, 10)),
             ReshapeLayer(rng, shape=(batchsize, 4*14*14), shape_inv=(batchsize, 4, 14, 14)),
             Pool2DLayer(rng, (batchsize, 4, 28, 28)),
             Conv2DLayer(rng, (4, 1, 5, 5), (batchsize, 1, 28, 28)),],

        rng=rng,
        # Noise std dev
        sigma=sigma
)


# Weights for the layer-wise unsupervised cost, as found in [1] given bottom-to-top,
# i.e. lambdas[0] is the weight of the reconstruction error for the first
# encoder and last decoder layer
lambdas = np.array([1000., 10., 0.1])

trainer = LadderAdamTrainer(rng=rng, batchsize=batchsize, epochs=2, alpha=0.0001, beta1=0.9, beta2=0.999, eps=1e-08, 
                            l1_weight=0.0, l2_weight=0.0, supervised_cost='cross_entropy')

trainer.train(network=network, lambdas=lambdas, labeled_train_input = l_train_set_x, labeled_train_output = l_train_set_y,
                                                unlabeled_train_input = u_train_set_x, unlabeled_train_output = u_train_set_y,
                                                valid_input = None, valid_output = None,
                                                test_input = None, test_output = None, filename=None)
