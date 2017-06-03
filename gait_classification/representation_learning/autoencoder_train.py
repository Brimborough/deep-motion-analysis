import numpy as np
import theano
import theano.tensor as T

from nn.AdamTrainer import AdamTrainer
from nn.ActivationLayer import ActivationLayer
from nn.AnimationPlotLines import animation_plot
from nn.DropoutLayer import DropoutLayer
from nn.Pool1DLayer import Pool1DLayer
from nn.Conv1DLayer import Conv1DLayer
from nn.ReshapeLayer import ReshapeLayer
from nn.HiddenLayer import HiddenLayer
from nn.LSTM1DLayer import LSTM1DLayer
from nn.Network import Network, AutoEncodingNetwork, InverseNetwork

from tools.utils import load_locomotion

rng = np.random.RandomState(23455)

shared = lambda d: theano.shared(d, borrow=True)
dataset, std, mean = load_locomotion(rng)

train_motion_dataset = dataset[0][0]

print "motion dataset shape = ", train_motion_dataset.shape

E = shared(train_motion_dataset)

BATCH_SIZE = 40

network = Network(
    DropoutLayer(rng, 0.2),
    Conv1DLayer(rng, (64, 66, 25), (BATCH_SIZE, 66, 240)),
    Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 240)),
    ActivationLayer(rng, f='elu'),

    DropoutLayer(rng, 0.2),
    Conv1DLayer(rng, (128, 64, 25), (BATCH_SIZE, 64, 120)),
    Pool1DLayer(rng, (2,), (BATCH_SIZE, 128, 120)),
    ActivationLayer(rng, f='elu'),
    
    InverseNetwork(Pool1DLayer(rng, (2,), (BATCH_SIZE, 128, 120))),
    DropoutLayer(rng, 0.2),  
    Conv1DLayer(rng, (64, 128, 25), (BATCH_SIZE, 128, 120)),
    ActivationLayer(rng, f='elu'),

    InverseNetwork(Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 240))),
    DropoutLayer(rng, 0.2),    
    Conv1DLayer(rng, (66, 64, 25), (BATCH_SIZE, 64, 240)),
    ActivationLayer(rng, f='elu'),
)

trainer = AdamTrainer(rng, batchsize=BATCH_SIZE, epochs=500, alpha=0.0005, cost='mse')
trainer.train(network, E, E, filename=[None, '../models/locomotion/ae/ae_layer_0.npz', None, None,
                                            None, '../models/locomotion/ae/ae_layer_1.npz', None, None,
                                            None, None, '../models/locomotion/ae/ae_layer_2.npz', None,
                                            None, None, '../models/locomotion/ae/ae_layer_3.npz', None,])

