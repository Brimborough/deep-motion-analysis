import numpy as np
import theano
import theano.tensor as T

from nn.ActivationLayer import ActivationLayer
from nn.NoiseLayer import NoiseLayer
from nn.Pool1DLayer import Pool1DLayer
from nn.Conv1DLayer import Conv1DLayer
from nn.Network import Network, AutoEncodingNetwork

rng = np.random.RandomState(23455)

BATCH_SIZE = 1
network = Network(

    NoiseLayer(rng, 0.3),
    
    Conv1DLayer(rng, (64, 66, 25), (BATCH_SIZE, 66, 240)),
    Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 240)),
    ActivationLayer(rng),

    Conv1DLayer(rng, (128, 64, 25), (BATCH_SIZE, 64, 120)),
    Pool1DLayer(rng, (2,), (BATCH_SIZE, 128, 120)),
    ActivationLayer(rng),
    
    Conv1DLayer(rng, (256, 128, 25), (BATCH_SIZE, 128, 60)),
    Pool1DLayer(rng, (2,), (BATCH_SIZE, 256, 60)),
    ActivationLayer(rng)
)
