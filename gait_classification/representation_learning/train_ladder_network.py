import numpy as np
import theano
import theano.tensor as T

from nn.ActivationLayer import ActivationLayer
from nn.AdamTrainer import AdamTrainer
from nn.BatchNormLayer import BatchNormLayer, InverseBatchNormLayer
from nn.HiddenLayer import HiddenLayer
from nn.LadderNetwork import LadderNetwork
from nn.NoiseLayer import GaussianNoiseLayer

from utils import load_data

rng = np.random.RandomState(23455)

dataset = '../data/mnist/mnist.pkl.gz'
datasets = load_data(dataset)

shared = lambda d: theano.shared(d, borrow=True)

train_set_x, train_set_y = datasets[0]
train_set_x = shared(train_set_x)

n_labelled_training_points = 1000

# Randomnly remove labels to create a semi-supervised setting
unlabelled_points = rng.random_integers(0, train_set_y.shape[0]-1, [train_set_y.shape[0]-n_labelled_training_points, 1])

# We mark unlabelled points by adding a vector of zeros instead of a one-hot vector
train_set_y[unlabelled_points] = 0

train_set_y = shared(train_set_y)

valid_set_x, valid_set_y = map(shared, datasets[1])
test_set_x, test_set_y = map(shared, datasets[2])

network = LadderNetwork(
        encoding_layers =
            [GaussianNoiseLayer(rng, sigma=1.0),
            HiddenLayer(rng, (784, 500)),
            BatchNormLayer((784, 500)),
            GaussianNoiseLayer(rng, sigma=1.0),
            ActivationLayer(rng, f='ReLU'),

            HiddenLayer(rng, (500, 10)),
            BatchNormLayer((500, 10)),
            GaussianNoiseLayer(rng, sigma=1.0),
            ActivationLayer(rng, f='softmax')],

        decoding_layers =
            [HiddenLayer(rng, (500, 10)),
            InverseBatchNormLayer((500, 10)),

            HiddenLayer(rng, (784, 500)),
            InverseBatchNormLayer((784, 500))]
)

trainer = AdamTrainer(rng=rng, batchsize=100, epochs=1, alpha=0.00001, cost='cross_entropy')
trainer.train(network=network, train_input=train_set_x, train_output=train_set_y,
                               valid_input=valid_set_x, valid_output=valid_set_y,
                               test_input=test_set_x, test_output=test_set_y, filename=None)
