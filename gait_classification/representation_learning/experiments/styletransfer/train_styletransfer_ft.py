import numpy as np
import sys
import theano
import theano.tensor as T

sys.path.append('../..')

from nn.ActivationLayer import ActivationLayer
from nn.AdamTrainer import AdamTrainer
from nn.BatchNormLayer import BatchNormLayer
from nn.Conv1DLayer import Conv1DLayer
from nn.DropoutLayer import DropoutLayer
from nn.HiddenLayer import HiddenLayer
from nn.Network import Network
from nn.Pool1DLayer import Pool1DLayer
from nn.ReshapeLayer import ReshapeLayer

from tools.utils import load_styletransfer

rng = np.random.RandomState(23455)

datasets = load_styletransfer(rng=rng, split=(0.6, 0.2, 0.2))

shared = lambda d: theano.shared(d, borrow=True)

train_set_x, train_set_y = map(shared, datasets[0])
valid_set_x, valid_set_y = map(shared, datasets[1])
test_set_x, test_set_y   = map(shared, datasets[2])

batchsize = 10

network = Network(
    Conv1DLayer(rng, (64, 66, 25), (batchsize, 66, 240)),
#    BatchNormLayer(rng, (batchsize, 64, 240)),
    ActivationLayer(rng, f='softplus'),
    Pool1DLayer(rng, (2,), (batchsize, 64, 240)),

    Conv1DLayer(rng, (128, 64, 25), (batchsize, 64, 120)),
#    BatchNormLayer(rng, (batchsize, 128, 120)),
    ActivationLayer(rng, f='softplus'),
    Pool1DLayer(rng, (2,), (batchsize, 128, 120)),
    
    Conv1DLayer(rng, (256, 128, 25), (batchsize, 128, 60)),
#    BatchNormLayer(rng, (batchsize, 256, 60)),
    ActivationLayer(rng, f='softplus'),
    Pool1DLayer(rng, (2,), (batchsize, 256, 60)),

    # Final pooling gives 1, 256, 30 

    # 256*60 = 7680
    ReshapeLayer(rng, (batchsize, 7680)),
    HiddenLayer(rng, (7680, 62)),
    ActivationLayer(rng, f='softmax'),
)

# Load the pre-trained conv-layers
network.load(['../models/cmu/dAe_v_0/layer_0.npz', None, None, # 1. Conv, BN, Activation, Pooling
              '../models/cmu/dAe_v_0/layer_1.npz', None, None, # 2. Conv, BN, Activation, Pooling
              '../models/cmu/dAe_v_0/layer_2.npz', None, None, # 3. Conv, BN, Activation, Pooling 
              None, None, None])                               # Reshape, Hidden, Activation

trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=30, alpha=0.00001, l1_weight=0.0, l2_weight=0.00001, cost='cross_entropy')

trainer.train(network=network, train_input=train_set_x, train_output=train_set_y,
              valid_input=valid_set_x, valid_output=valid_set_y, 
              filename=['../models/styletransfer/pretrained_v_0/layer_0.npz',  None, None,       # 1. Conv, BN, Activation, Pooling
                        '../models/styletransfer/pretrained_v_0/layer_1.npz',  None, None,       # 2. Conv, BN, Activation, Pooling
                        '../models/styletransfer/pretrained_v_0/layer_2.npz',  None, None,       # 3. Conv, BN, Activation, Pooling
                        None, '../models/styletransfer/pretrained_v_0/layer_3.npz', None, None]) # Reshape, Hiden, Activation

trainer.eval(network=network, eval_input=test_set_x, eval_output=test_set_y, filename=None)
