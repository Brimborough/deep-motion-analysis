import numpy as np
import theano
import theano.tensor as T

from nn.ActivationLayer import ActivationLayer
from nn.AdamTrainer import LadderAdamTrainer
from nn.BatchNormLayer import BatchNormLayer, InverseBatchNormLayer
from nn.HiddenLayer import HiddenLayer
from nn.LadderNetwork import LadderNetwork
from nn.NoiseLayer import GaussianNoiseLayer

from utils import load_data, remove_labels

rng = np.random.RandomState(23455)

dataset = '../data/mnist/mnist.pkl.gz'
datasets = load_data(dataset)

shared = lambda d: theano.shared(d, borrow=True)

train_set_x, train_set_y = datasets[0]
train_set_x = shared(train_set_x)

# Create a semi-supervised task by removing all but a thousand labels
#train_set_y = remove_labels(rng, train_set_y, 1000)
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
            InverseBatchNormLayer((10, 500)),

            HiddenLayer(rng, (784, 500)),
            InverseBatchNormLayer((500, 784))]
)

# Weights for the layer-wise unsupervised cost, as found in [1] given bottom-to-top, 
# i.e. lambdas[0] is the weight of the reconstruction error for the first
# encoder and last decoder layer
#
# [1] Rasmus, Antti, et al. "Semi-Supervised Learning with Ladder Networks." 
# Advances in Neural Information Processing Systems. 2015."""
lambdas = np.array([1., 1., 1.])

trainer = LadderAdamTrainer(rng=rng, batchsize=1, epochs=1, alpha=0.00001, beta1=0.9, beta2=0.999, eps=1e-08,
                            l1_weight=0.0, l2_weight=0.0, supervised_cost='cross_entropy')

trainer.train(network=network, lambdas=lambdas, train_input=train_set_x, train_output=train_set_y,
                                                valid_input=valid_set_x, valid_output=valid_set_y,
                                                test_input=test_set_x, test_output=test_set_y, filename=None)
