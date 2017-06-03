import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T

from nn.ActivationLayer import ActivationLayer
from nn.HiddenLayer import HiddenLayer
from nn.Conv1DLayer import Conv1DLayer
from nn.Pool1DLayer import Pool1DLayer
from nn.NoiseLayer import NoiseLayer
from nn.BatchNormLayer import BatchNormLayer
from nn.DropoutLayer import DropoutLayer
from nn.VariationalLayer import VariationalLayer
from nn.Network import Network, AutoEncodingNetwork, InverseNetwork
from nn.AdversarialVaeAdamTrainer import AdversarialVaeAdamTrainer
from nn.ReshapeLayer import ReshapeLayer

from nn.AnimationPlotLines import animation_plot

from tools.utils import load_locomotion

rng = np.random.RandomState(23455)

shared = lambda d: theano.shared(d, borrow=True)

dataset, std, mean = load_locomotion(rng)
E = shared(dataset[0][0])

BATCH_SIZE = 40

FC_SIZE = 800

encoderNetwork = Network(
    Conv1DLayer(rng, (64, 66, 25), (BATCH_SIZE, 66, 240)),
    BatchNormLayer(rng, (BATCH_SIZE, 64, 240)),
    ActivationLayer(rng, f='elu'),
    Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 240)),

    DropoutLayer(rng, 0.25),
    Conv1DLayer(rng, (128, 64, 25), (BATCH_SIZE, 64, 120)),
    BatchNormLayer(rng, (BATCH_SIZE, 128, 120)),
    ActivationLayer(rng, f='elu'),
    Pool1DLayer(rng, (2,), (BATCH_SIZE, 128, 120)),

    ReshapeLayer(rng, (BATCH_SIZE, 128*60)),
    DropoutLayer(rng, 0.25),    
    HiddenLayer(rng, (128*60, FC_SIZE)),
    BatchNormLayer(rng, (128*60, FC_SIZE)),
    ActivationLayer(rng, f='elu'),
)

variationalNetwork = Network(
    VariationalLayer(rng),
)

decoderNetwork = Network(
    HiddenLayer(rng, (FC_SIZE/2, 64*30)),
    BatchNormLayer(rng, (FC_SIZE/2, 64*30)),
    ActivationLayer(rng, f='elu'),
    ReshapeLayer(rng, (BATCH_SIZE, 64, 30)),

    InverseNetwork(Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 60))),
    DropoutLayer(rng, 0.15),    
    Conv1DLayer(rng, (64, 64, 25), (BATCH_SIZE, 64, 60)),
    ActivationLayer(rng, f='elu'),

    InverseNetwork(Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 120))),
    DropoutLayer(rng, 0.25),  
    Conv1DLayer(rng, (64, 64, 25), (BATCH_SIZE, 64, 120)),
    ActivationLayer(rng, f='elu'),

    InverseNetwork(Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 240))),
    DropoutLayer(rng, 0.25),    
    Conv1DLayer(rng, (66, 64, 25), (BATCH_SIZE, 64, 240)),
    ActivationLayer(rng, f='elu'),
)

discriminatorNetwork = Network(
    Conv1DLayer(rng, (64, 66, 25), (BATCH_SIZE, 66, 240)),
    Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 240)),
    ActivationLayer(rng, f='elu'),

    DropoutLayer(rng, 0.25),
    Conv1DLayer(rng, (128, 64, 25), (BATCH_SIZE, 64, 120)),
    Pool1DLayer(rng, (2,), (BATCH_SIZE, 128, 120)),
    ActivationLayer(rng, f='elu'),

    ReshapeLayer(rng, (BATCH_SIZE, 128*60)),
    DropoutLayer(rng, 0.25),    
    HiddenLayer(rng, (128*60, 1600)),
    ActivationLayer(rng, f='elu'),

    DropoutLayer(rng, 0.2),
    HiddenLayer(rng, (1600, 1)),
    BatchNormLayer(rng, (1600, 1)),
    ActivationLayer(rng, f='sigmoid'),  
)

def encoder_cost(enc_out, dec_out, input_data):
    vari_amount = 1.0
    repr_amount = 1.0
    
    H = enc_out
    mu, sg = H[:,0::2], H[:,1::2]

    vari_cost = -0.5 * T.mean(1 + 2 * sg - mu**2 - T.exp(2 * sg))
    repr_cost = T.mean((dec_out - input_data)**2)
    
    return repr_amount * repr_cost + vari_amount * vari_cost, vari_cost, repr_cost

def decoder_cost(dec_real_out, disc_sample_out, disc_fake_out, disc_real_out, input_data):
    disc_cost = T.nnet.binary_crossentropy(disc_real_out, np.zeros(1, dtype=theano.config.floatX)).mean()
    disc_cost = T.nnet.binary_crossentropy(disc_fake_out, np.ones(1, dtype=theano.config.floatX)).mean() 
    disc_cost += T.nnet.binary_crossentropy(disc_sample_out, np.zeros(1, dtype=theano.config.floatX)).mean()

    reconst_cost = T.mean((dec_real_out - input_data)**2)

    disc_weight = 1
    recons_weight = 100

    return (disc_weight * disc_cost + recons_weight * reconst_cost)

def discriminative_cost(disc_sample_out, disc_fake_out, disc_real_out):
    disc_cost = T.nnet.binary_crossentropy(disc_real_out, np.ones(1, dtype=theano.config.floatX)).mean()
    disc_cost += T.nnet.binary_crossentropy(disc_fake_out, np.zeros(1, dtype=theano.config.floatX)).mean()
    disc_cost += T.nnet.binary_crossentropy(disc_sample_out, np.ones(1, dtype=theano.config.floatX)).mean()
    
    return disc_cost

trainer = AdversarialVaeAdamTrainer(rng=rng, 
                                batchsize=BATCH_SIZE, 
                                enc_cost=encoder_cost, 
                                dec_cost=decoder_cost,
                                disc_cost=discriminative_cost,
                                epochs=200)

trainer.train(enc_network=encoderNetwork, 
                                dec_network=decoderNetwork,
                                disc_network=discriminatorNetwork,
                                var_network=variationalNetwork,
                                train_input=E,
                                filename=['../models/locomotion/adv_vae/v_3/layer_0.npz',
                                        '../models/locomotion/adv_vae/v_3/layer_1.npz', 
                                        None, None,
                                        None, None, '../models/locomotion/adv_vae/v_3/layer_2.npz', None,
                                        None, None, '../models/locomotion/adv_vae/v_3/layer_3.npz', None,
                                        None, None, '../models/locomotion/adv_vae/v_3/layer_4.npz', None,])

#Testing
BATCH_SIZE = 100

generatorNetwork = Network(
    HiddenLayer(rng, (FC_SIZE/2, 64*30)),
    BatchNormLayer(rng, (FC_SIZE/2, 64*30)),
    ActivationLayer(rng, f='elu'),
    ReshapeLayer(rng, (BATCH_SIZE, 64, 30)),

    InverseNetwork(Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 60))),
    DropoutLayer(rng, 0.15),    
    Conv1DLayer(rng, (64, 64, 25), (BATCH_SIZE, 64, 60)),
    ActivationLayer(rng, f='elu'),

    InverseNetwork(Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 120))),
    DropoutLayer(rng, 0.25),  
    Conv1DLayer(rng, (64, 64, 25), (BATCH_SIZE, 64, 120)),
    ActivationLayer(rng, f='elu'),

    InverseNetwork(Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 240))),
    DropoutLayer(rng, 0.25),    
    Conv1DLayer(rng, (66, 64, 25), (BATCH_SIZE, 64, 240)),
    ActivationLayer(rng, f='elu'),
)

generatorNetwork.load(['../models/locomotion/adv_vae/v_3/layer_0.npz',
                                        '../models/locomotion/adv_vae/v_3/layer_1.npz', 
                                        None, None,
                                        None, None, '../models/locomotion/adv_vae/v_3/layer_2.npz', None,
                                        None, None, '../models/locomotion/adv_vae/v_3/layer_3.npz', None,
                                        None, None, '../models/locomotion/adv_vae/v_3/layer_4.npz', None,])

def randomize_uniform_data(n_input):
    return rng.uniform(size=(n_input, FC_SIZE/2), 
            low=-np.sqrt(3, dtype=theano.config.floatX), 
            high=np.sqrt(3, dtype=theano.config.floatX)).astype(theano.config.floatX)

gen_rand_input = theano.shared(randomize_uniform_data(100), name = 'z')
generate_sample_motions = theano.function([], generatorNetwork(gen_rand_input))
sample = generate_sample_motions()

result = sample * (std + 1e-10) + mean

new1 = result[25:26]
new2 = result[26:27]
new3 = result[0:1]

animation_plot([new1, new2, new3], filename='vae-gan-cmu.mp4',interval=15.15)
