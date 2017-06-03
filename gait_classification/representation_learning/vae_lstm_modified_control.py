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
from nn.VariationalConvLayer import VariationalConvLayer
from nn.LSTM1DTestLayer import LSTM1DTestLayer
from nn.Network import Network, AutoEncodingNetwork, InverseNetwork

from tools.utils import load_locomotion

rng = np.random.RandomState(23455)

shared = lambda d: theano.shared(d, borrow=True)
dataset, std, mean = load_locomotion(rng)
train_dataset = dataset[0][0][:100]
train_dataset = train_dataset.swapaxes(0, 1)[-3:]
train_dataset = train_dataset.swapaxes(0, 1)
train_dataset = train_dataset[:100]
print "train_dataset.shape = ", train_dataset.shape

E = shared(train_dataset)

train_dataset[:][:][1] = 0

for i in xrange(train_dataset.shape[0]):
    train_dataset[i][:][1] = 0

train_hidden = np.load("vae_encoder_hidden_space_cmu.npz")
train_hidden = train_hidden['hidden_space']

Xmean = train_hidden.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]
Xstd = np.array([[[train_hidden.std()]]]).repeat(train_hidden.shape[1], axis=1)
train_hidden = (train_hidden - Xmean) / Xstd

train_hidden = train_hidden[:100]

H = shared(train_hidden)

print "train_hidden.shape = ", train_hidden.shape

BATCH_SIZE = 100
H_SIZE = 128

encoder = HiddenLayer(rng, (128, H_SIZE))
encode_igate = HiddenLayer(rng, (128, H_SIZE))
encode_fgate = HiddenLayer(rng, (128, H_SIZE))

recoder = HiddenLayer(rng, (H_SIZE, H_SIZE))
recode_igate = HiddenLayer(rng, (H_SIZE, H_SIZE))
recode_fgate = HiddenLayer(rng, (H_SIZE, H_SIZE))

activation = ActivationLayer(rng, f='elu')
dropout = DropoutLayer(rng, 0.2)

control_network = Network(    
    Conv1DLayer(rng, (64, 3, 25), (BATCH_SIZE, 3, 240)),
    Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 240)),
    ActivationLayer(rng, f='elu'),

    Conv1DLayer(rng, (128, 64, 25), (BATCH_SIZE, 64, 120)),
    Pool1DLayer(rng, (2,), (BATCH_SIZE, 128, 120)),
    ActivationLayer(rng, f='elu'),

    LSTM1DTestLayer(encoder, recoder, encode_igate, recode_igate, encode_fgate, recode_fgate, activation, dropout, H),
)

control_network.load(['../models/cmu/vae_lstm/1_normalized_layer_0.npz', None, None, 
                '../models/cmu/vae_lstm/1_normalized_layer_1.npz', None, None, 
                '../models/cmu/vae_lstm/1_normalized_layer_2.npz',])

control_func = theano.function([], control_network(E))
control_result = control_func()
print control_result.shape

C = shared(control_result)

var_network = Network(
    VariationalConvLayer(rng, sample=False),
)

var_func = theano.function([], var_network(C))
var_result = var_func()
print var_result.shape

V = shared(var_result)

decoder_network = Network(
    InverseNetwork(Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 120))),
    Conv1DLayer(rng, (64, 64, 25), (BATCH_SIZE, 64, 120)),
    ActivationLayer(rng, f='elu'),

    InverseNetwork(Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 240))),
    Conv1DLayer(rng, (66, 64, 25), (BATCH_SIZE, 64, 240)),
    ActivationLayer(rng, f='elu'),
)

decoder_network.load([None, '../models/cmu/vae/l_2_mo4.npz', None,
                        None, '../models/cmu/vae/l_3_mo4.npz', None,])

decoder_func = theano.function([], decoder_network(V))
decoder_result = decoder_func()
print decoder_result.shape

np.savez_compressed("decoder_result_modified_control_normalized_init_2.npz", decoder_result=decoder_result)
