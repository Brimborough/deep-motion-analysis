# Implementation of Variational Autoencoder (VAE) for human motion synthesis

import numpy as np
import theano
import theano.tensor as T

from nn.OriAdamTrainer import AdamTrainer
from nn.ActivationLayer import ActivationLayer
from nn.AnimationPlotLines import animation_plot
from nn.DropoutLayer import DropoutLayer
from nn.Pool1DLayer import Pool1DLayer
from nn.Conv1DLayer import Conv1DLayer
from nn.ReshapeLayer import ReshapeLayer
from nn.HiddenLayer import HiddenLayer
from nn.BatchNormLayer import BatchNormLayer
from nn.VariationalConvLayer import VariationalConvLayer
from nn.Network import Network, AutoEncodingNetwork, InverseNetwork

from tools.utils import load_locomotion

rng = np.random.RandomState(23455)

BATCH_SIZE = 40
FC_SIZE = 800

shared = lambda d: theano.shared(d, borrow=True)
dataset, std, mean = load_locomotion(rng)
E = shared(dataset[0][0])

network = Network(
    
    Network(
        DropoutLayer(rng, 0.2),
        Conv1DLayer(rng, (64, 66, 25), (BATCH_SIZE, 66, 240)),
        Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 240)),
        ActivationLayer(rng, f='elu'),

        DropoutLayer(rng, 0.2),
        Conv1DLayer(rng, (128, 64, 25), (BATCH_SIZE, 64, 120)),
        Pool1DLayer(rng, (2,), (BATCH_SIZE, 128, 120)),
        ActivationLayer(rng, f='elu'),
    ),
    
    Network(
        VariationalConvLayer(rng),
    ),
    
    Network(
        InverseNetwork(Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 120))),
        DropoutLayer(rng, 0.2),  
        Conv1DLayer(rng, (64, 64, 25), (BATCH_SIZE, 64, 120)),
        ActivationLayer(rng, f='elu'),

        InverseNetwork(Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 240))),
        DropoutLayer(rng, 0.2),    
        Conv1DLayer(rng, (66, 64, 25), (BATCH_SIZE, 64, 240)),
        ActivationLayer(rng, f='elu'),
    )
)

def cost(networks, X, Y):
    network_u, network_v, network_d = networks.layers
    
    vari_amount = 1.0
    repr_amount = 1.0
    
    H = network_u(X)
    mu, sg = H[:, :, 0::2], H[:, :, 1::2]
       
    #ya0st VAE
    vari_cost = 0.5 * T.mean(1 + 2 * sg - mu**2 - T.exp(2 * sg))
    repr_cost = T.mean((network_d(network_v(H)) - Y)**2)

    return repr_amount * repr_cost - vari_amount * vari_cost


trainer = AdamTrainer(rng, batchsize=BATCH_SIZE, epochs=250, alpha=0.00005, cost=cost)
trainer.train(network, E, E, filename=[[None, '../models/locomotion/vae/l_0_mo4.npz', None, None, 
                                        None, '../models/locomotion/vae/l_1_mo4.npz', None, None,],
                                        [None,],
                                        [None, None, '../models/locomotion/vae/l_2_mo4.npz', None,
                                        None, None, '../models/locomotion/vae/l_3_mo4.npz', None,],])
