import numpy as np
import sys
import theano
import theano.tensor as T

sys.path.append('../..')

from nn.ActivationLayer import ActivationLayer
from nn.AdamTrainer import AdamTrainer
from nn.BatchNormLayer import BatchNormLayer
from nn.Conv1DLayer import Conv1DLayer
from nn.HiddenLayer import HiddenLayer
from nn.Network import Network
from nn.Pool1DLayer import Pool1DLayer
from nn.ReshapeLayer import ReshapeLayer

from tools.utils import fair_split, load_styletransfer

rng = np.random.RandomState(23455)


datasets = load_styletransfer(rng, (0.6, 0.2, 0.2))

shared = lambda d: theano.shared(d, borrow=True)

train_set_x, train_set_y = map(shared, datasets[0])
valid_set_x, valid_set_y = map(shared, datasets[1])
test_set_x, test_set_y   = map(shared, datasets[2])

batchsize = 2

network = Network(
    Conv1DLayer(rng, (64, 66, 25), (batchsize, 66, 240)),
    BatchNormLayer(rng, (batchsize, 64, 240)),
    ActivationLayer(rng, f='ReLU'),
    Pool1DLayer(rng, (2,), (batchsize, 64, 240)),

    Conv1DLayer(rng, (128, 64, 25), (batchsize, 64, 120)),
    BatchNormLayer(rng, (batchsize, 128, 120)),
    ActivationLayer(rng, f='ReLU'),
    Pool1DLayer(rng, (2,), (batchsize, 128, 120)),
    
    Conv1DLayer(rng, (256, 128, 25), (batchsize, 128, 60)),
    BatchNormLayer(rng, (batchsize, 256, 60)),
    ActivationLayer(rng, f='ReLU'),
    Pool1DLayer(rng, (2,), (batchsize, 256, 60)),

    # Final pooling gives 1, 256, 30 

    # 256*60 = 7680
    ReshapeLayer(rng, (batchsize, 7680)),
    HiddenLayer(rng, (np.prod([256, 30]), 8)),
    ActivationLayer(rng, f='softmax'),
)

# Load the pre-trained conv-layers
#network.load(['../models/conv_ae/layer_0.npz', None, None,
#              '../models/conv_ae/layer_1.npz', None, None,
#              '../models/conv_ae/layer_2.npz', None, None, 
#              None, None, None])

trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=1, alpha=0.00001, cost='cross_entropy')

# Fine-tuning for classification
trainer.train(network=network, train_input=train_set_x, train_output=train_set_y, 
                               valid_input=valid_set_x, valid_output=valid_set_y,
                               test_input=test_set_x, test_output=test_set_y,
                               # Don't save the params
                               filename=None)
