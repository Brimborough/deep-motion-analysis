import numpy as np
import theano
import theano.tensor as T

from nn.AdamTrainer import AdamTrainer
from nn.Network import AutoEncodingNetwork

from utils import load_data

from nn.ActivationLayer import ActivationLayer
from nn.AnimationPlotLines import animation_plot
from nn.DropoutLayer import DropoutLayer
from nn.Pool2DLayer import Pool2DLayer
from nn.Conv2DLayer import Conv2DLayer
from nn.VariationalLayer import VariationalLayer
from nn.Network import Network, AutoEncodingNetwork, InverseNetwork
import matplotlib.pyplot as plt

rng = np.random.RandomState(23455)

dataset = '../data/mnist/mnist.pkl.gz'
datasets = load_data(dataset)

shared = lambda d: theano.shared(d, borrow=True)

train_set_x, train_set_y = map(shared, datasets[0])
valid_set_x, valid_set_y = map(shared, datasets[1])
test_set_x, test_set_y   = map(shared, datasets[2])

train_set_x = train_set_x.reshape((50000, 1, 28, 28))
valid_set_x = valid_set_x.reshape((10000, 1, 28, 28))
test_set_x  = test_set_x.reshape((10000, 1, 28, 28))

batchsize = 1
network = Network(
    
    Network(
        DropoutLayer(rng, 0.25),
        Conv2DLayer(rng, (64, 1, 5, 5), (batchsize, 1, 28, 28)),
        ActivationLayer(rng, f='ReLU'),
        Pool2DLayer(rng, (batchsize, 64, 28, 28)),

        DropoutLayer(rng, 0.25),    
        Conv2DLayer(rng, (128, 64, 5, 5), (batchsize, 64, 14, 14)),
        ActivationLayer(rng, f='ReLU'),
        Pool2DLayer(rng, (batchsize, 128, 14, 14)),
    ),
    
    Network(
        VariationalLayer(rng),
    ),
    
    Network(
        InverseNetwork(Pool2DLayer(rng, (batchsize, 64, 14, 14))),
        DropoutLayer(rng, 0.25),
        Conv2DLayer(rng, (32, 64, 5, 5), (batchsize, 64, 14, 14)),
        ActivationLayer(rng, f='ReLU'),

        InverseNetwork(Pool2DLayer(rng, (batchsize, 32, 28, 28))),
        DropoutLayer(rng, 0.25),
        Conv2DLayer(rng, (1, 32, 5, 5), (batchsize, 32, 28, 28)),
        ActivationLayer(rng, f='ReLU'),
    )
)

def cost(networks, X, Y):
    network_u, network_v, network_d = networks.layers
    
    vari_amount = 1.0
    repr_amount = 1.0
    
    H = network_u(X)

    mu, sg = H[:,0::2], H[:,1::2]
    
    vari_cost = 0.5 * T.mean(mu**2) + 0.5 * T.mean((T.sqrt(T.exp(sg))-1)**2)
    repr_cost = T.mean((network_d(network_v(H)) - Y)**2)
    
    return repr_amount * repr_cost + vari_amount * vari_cost
    #return vari_amount * vari_cost

print "VAE MNIST"

trainer = AdamTrainer(rng, batchsize=1, epochs=50, alpha=0.00001, cost=cost)
trainer.train(network, train_set_x, train_set_x, filename=[[None, '../models/mnist/conv_varae/v_1/layer_0.npz', None, None, 
                                        None, '../models/mnist/conv_varae/v_1/layer_1.npz', None, None,],
                                        [None,],
                                        [None, None, '../models/mnist/conv_varae/v_1/layer_2.npz', None,
                                        None, None, '../models/mnist/conv_varae/v_1/layer_3.npz', None],])

result = trainer.get_representation(network, train_set_x, 2)

sample = result[:100]
sample = sample.reshape((10,10,28,28)).transpose(1,2,0,3).reshape((10*28, 10*28))
plt.imshow(sample, cmap = plt.get_cmap('gray'), vmin=0, vmax=1)
plt.savefig('samples/sampleImages_new')

