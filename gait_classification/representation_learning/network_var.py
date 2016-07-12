import numpy as np
import theano
import theano.tensor as T

from nn.ActivationLayer import ActivationLayer
from nn.DropoutLayer import DropoutLayer
from nn.Pool1DLayer import Pool1DLayer
from nn.Conv1DLayer import Conv1DLayer
from nn.VariationalLayer import VariationalLayer
from nn.Network import Network, AutoEncodingNetwork, InverseNetwork

rng = np.random.RandomState(23455)

BATCH_SIZE = 1
network = Network(
    
    Network(
        #DropoutLayer(rng, 0.25),    
        Conv1DLayer(rng, (64, 66, 25), (BATCH_SIZE, 66, 240)),
        Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 240)),
        ActivationLayer(rng),

        DropoutLayer(rng, 0.25),    
        Conv1DLayer(rng, (128, 64, 25), (BATCH_SIZE, 64, 120)),
        Pool1DLayer(rng, (2,), (BATCH_SIZE, 128, 120)),
        ActivationLayer(rng),
    ),
    
    Network(
        VariationalLayer(rng),
    ),
    
    Network(
        InverseNetwork(Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 120))),
        DropoutLayer(rng, 0.25),    
        Conv1DLayer(rng, (64, 64, 25), (BATCH_SIZE, 64, 120)),
        ActivationLayer(rng),

        InverseNetwork(Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 240))),
        DropoutLayer(rng, 0.25),    
        Conv1DLayer(rng, (66, 64, 25), (BATCH_SIZE, 64, 240)),
    )
)
