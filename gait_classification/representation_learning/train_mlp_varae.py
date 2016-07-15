import numpy as np
import theano
import theano.tensor as T

from nn.ActivationLayer import ActivationLayer
from nn.BatchNormLayer import BatchNormLayer
from nn.HiddenLayer import HiddenLayer
from nn.Network import Network
from nn.AdamTrainer import AdamTrainer
from nn.VariationalLayerKeras import VariationalLayer

from tools.utils import load_mnist

rng = np.random.RandomState(23455)

datasets = load_mnist(rng)

shared = lambda d: theano.shared(d, borrow=True)

train_set_x, train_set_y = map(shared, datasets[0])
valid_set_x, valid_set_y = map(shared, datasets[1])
test_set_x, test_set_y   = map(shared, datasets[2])

network = Network(
	Network(
	    HiddenLayer(rng, (784, 256)),
	    ActivationLayer(rng, f='ReLU'),

	    HiddenLayer(rng, (256, 64)),
	    ActivationLayer(rng, f='ReLU'),
	),

	Network(
        VariationalLayer(rng),
    ),

	Network(
	    HiddenLayer(rng, (32, 64)),
    	ActivationLayer(rng, f='ReLU'),
    	HiddenLayer(rng, (64, 256)),
    	ActivationLayer(rng, f='ReLU'),
    	HiddenLayer(rng, (256, 784)),
    	ActivationLayer(rng, f='sigmoid'),
    )
)

def cost(networks, X, Y):
    network_u, network_v, network_d = networks.layers
    
    vari_amount = 1.0
    repr_amount = 1.0
    
    H = network_u(X)
    mu, sg = H[:,0::2], H[:,1::2]

    repr_cost = T.mean((network_d(network_v(H)) - Y)**2)
    vari_cost = -0.5 + T.mean(1 + sg - T.sqrt(mu) - T.exp(sg))

    return repr_amount * repr_cost + vari_amount * vari_cost

trainer = AdamTrainer(rng=rng, batchsize=25, epochs=40, alpha=0.00001, cost=cost)
#trainer.train(network=network, train_input=train_set_x, train_output=train_set_x, filename=None)

result = trainer.get_representation(network, train_set_x, 2)  * (std + 1e-10) + mean

print result.shape

"""
sample = sample[0:100].reshape((10,10,28,28)).transpose(1,2,0,3).reshape((10*28, 10*28))
plt.imshow(sample, cmap = plt.get_cmap('gray'), vmin=0, vmax=1)
plt.savefig('vae_samples/sampleImages_mlp')
"""