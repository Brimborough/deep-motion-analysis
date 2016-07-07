import numpy as np
import theano
import theano.tensor as T

from nn.ActivationLayer import ActivationLayer
from nn.BinaryTaskTrainer import BinaryTaskTrainer
from nn.AdamTrainer import AdamTrainer
from nn.BatchNormLayer import BatchNormLayer
from nn.Conv1DLayer import Conv1DLayer
from nn.HiddenLayer import HiddenLayer
from nn.MultiTaskLayer import MultiTaskLayer
from nn.Network import MultiTaskNetwork, InverseNetwork, Network
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

classification_network = Network(
    HiddenLayer(rng, (250, 62)),
    ActivationLayer(rng, f='softmax'),
)

reconstruction_network = InverseNetwork(Network(
    Conv1DLayer(rng, (64, 66, 25), (batchsize, 66, 240)),
    ActivationLayer(rng, f='ReLU'),
    Pool1DLayer(rng, (2,), (batchsize, 64, 240)),

    Conv1DLayer(rng, (128, 64, 25), (batchsize, 64, 120)),
    ActivationLayer(rng, f='ReLU'),
    Pool1DLayer(rng, (2,), (batchsize, 128, 120)),
    
    Conv1DLayer(rng, (256, 128, 25), (batchsize, 128, 60)),
    ActivationLayer(rng, f='ReLU'),
    Pool1DLayer(rng, (2,), (batchsize, 256, 60)),

    # Final pooling gives 1, 256, 30 

    # 256*60 = 7680
    ReshapeLayer(rng, shape=(batchsize, 7680), shape_inv=(batchsize, 256, 30)),
    HiddenLayer(rng, (7680, 250)),
    ActivationLayer(rng, f='ReLU')
))

network = MultiTaskNetwork(
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
    HiddenLayer(rng, (7680, 250)),
    ActivationLayer(rng, f='ReLU'),

    MultiTaskLayer(classification_network, 
                   reconstruction_network)
)

# Load the pre-trained conv-layers
#network.load(['../models/conv_ae/layer_0.npz', None, None,
#              '../models/conv_ae/layer_1.npz', None, None,
#              '../models/conv_ae/layer_2.npz', None, None, 
#              None, None, None])

#trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=1, alpha=0.00001, cost='cross_entropy')
#
#trainer.train(network=network, train_input=train_set_x, train_output=train_set_y,
#                               valid_input=valid_set_x, valid_output=valid_set_y,
#                               filename=None, logging=True)

trainer = BinaryTaskTrainer(rng=rng, batchsize=batchsize, epochs=1, alpha=0.001, costs=['cross_entropy', 'mse'])

trainer.train(network=network, train_input=train_set_x, train_outputs=[train_set_y, train_set_x], 
              valid_input=valid_set_x, valid_output=valid_set_y,
              filename=['../models/styletransfer/hybrid_v_0/layer_0.npz', '../models/styletransfer/hybrid_v_0/bn_0.npz', None, None, # 1. Conv, BN, Activation, Pooling
                        '../models/styletransfer/hybrid_v_0/layer_1.npz', '../models/styletransfer/hybrid_v_0/bn_1.npz', None, None, # 2. Conv, BN, Activation, Pooling
                        '../models/styletransfer/hybrid_v_0/layer_2.npz', '../models/styletransfer/hybrid_v_0/bn_2.npz', None, None, # 3. Conv, BN, Activation, Pooling
                        None, '../models/styletransfer/hybrid_v_0/layer_3.npz', None, None])                                         # Reshape, Hiden, Activation

trainer.eval(network=network, eval_input=test_set_x, eval_output=test_set_y, filename=None)
