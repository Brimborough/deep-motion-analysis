import numpy as np
import theano
import theano.tensor as T

from nn.FullyTrainedVaeLstmAdamTrainer import AdamTrainer
from nn.ActivationLayer import ActivationLayer
from nn.AnimationPlotLines import animation_plot
from nn.DropoutLayer import DropoutLayer
from nn.Pool1DLayer import Pool1DLayer
from nn.Conv1DLayer import Conv1DLayer
from nn.ReshapeLayer import ReshapeLayer
from nn.HiddenLayer import HiddenLayer
from nn.VariationalLayerUnflattened import VariationalLayer
from nn.LSTM1DHiddenInitLayer import LSTM1DLayer
from nn.Network import Network, AutoEncodingNetwork, InverseNetwork

from tools.utils import load_locomotion
from tools.utils import load_terrain

rng = np.random.RandomState(23455)

shared = lambda d: theano.shared(d, borrow=True)
dataset1, std1, mean1 = load_locomotion(rng)
dataset2, std2, mean2 = load_terrain(rng)

dataset = np.concatenate([dataset1[0][0][:300], dataset2[0][0]], axis=0)

train_control_dataset = dataset[:1800]
train_control_dataset = train_control_dataset.swapaxes(0, 1)[-3:]
train_control_dataset = train_control_dataset.swapaxes(0, 1)

print "control dataset shape = ", train_control_dataset.shape

E = shared(train_control_dataset)

dataset1, std1, mean1 = load_locomotion(rng)
dataset2, std2, mean2 = load_terrain(rng)

dataset = np.concatenate([dataset1[0][0][:300], dataset2[0][0]], axis=0)

train_motion_input_dataset = dataset[:1800]
train_motion_input_dataset = train_motion_input_dataset.swapaxes(0, 1)[:-3]
train_motion_input_dataset = train_motion_input_dataset.swapaxes(0, 1)

print "motion input dataset shape = ", train_motion_input_dataset.shape

M_I = shared(train_motion_input_dataset)

dataset1, std1, mean1 = load_locomotion(rng)
dataset2, std2, mean2 = load_terrain(rng)

dataset = np.concatenate([dataset1[0][0][:300], dataset2[0][0]], axis=0)

train_motion_output_dataset = dataset[:1800]

print "motion output dataset shape = ", train_motion_output_dataset.shape
M_O = shared(train_motion_output_dataset)

BATCH_SIZE = 40
H_SIZE = 128

encoder = HiddenLayer(rng, (128, H_SIZE))
encode_igate = HiddenLayer(rng, (128, H_SIZE))
encode_fgate = HiddenLayer(rng, (128, H_SIZE))

recoder = HiddenLayer(rng, (H_SIZE, H_SIZE))
recode_igate = HiddenLayer(rng, (H_SIZE, H_SIZE))
recode_fgate = HiddenLayer(rng, (H_SIZE, H_SIZE))

activation = ActivationLayer(rng, f='elu')
dropout = DropoutLayer(rng, 0.2)

network = Network(
    Network(
        DropoutLayer(rng, 0.2),
        Conv1DLayer(rng, (64, 63, 25), (BATCH_SIZE, 63, 240)),
        Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 240)),
        ActivationLayer(rng, f='elu'),

        DropoutLayer(rng, 0.2),
        Conv1DLayer(rng, (256, 64, 25), (BATCH_SIZE, 64, 120)),
        Pool1DLayer(rng, (2,), (BATCH_SIZE, 256, 120)),
        ActivationLayer(rng, f='elu'),
    ),
    
    Network(
        VariationalLayer(rng),
    ),
    
    Network(
        InverseNetwork(Pool1DLayer(rng, (2,), (BATCH_SIZE, 128, 120))),
        DropoutLayer(rng, 0.2),  
        Conv1DLayer(rng, (128, 128, 25), (BATCH_SIZE, 128, 120)),
        ActivationLayer(rng, f='elu'),

        InverseNetwork(Pool1DLayer(rng, (2,), (BATCH_SIZE, 128, 240))),
        DropoutLayer(rng, 0.2),    
        Conv1DLayer(rng, (66, 128, 25), (BATCH_SIZE, 128, 240)),
        ActivationLayer(rng, f='elu'),
    ),

    Network(
        DropoutLayer(rng, 0.2),
        Conv1DLayer(rng, (64, 3, 25), (BATCH_SIZE, 3, 240)),
        Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 240)),
        ActivationLayer(rng, f='elu'),

        DropoutLayer(rng, 0.2),
        Conv1DLayer(rng, (128, 64, 25), (BATCH_SIZE, 64, 120)),
        Pool1DLayer(rng, (2,), (BATCH_SIZE, 128, 120)),
        ActivationLayer(rng, f='elu'),
    ),

    Network(
        LSTM1DLayer(encoder, recoder, encode_igate, recode_igate, encode_fgate, recode_fgate, activation, dropout),
    )
)

def cost(networks, X, W, Y):
    n_enc, n_var, n_dec, n_ff, n_lstm = networks.layers
    
    H = n_enc(X)
    mu, sg = H[:, 0::2, :], H[:, 1::2, :]
    
    vari_cost = -0.5 * T.mean(1 + 2 * sg - mu**2 - T.exp(2 * sg))

    F = n_ff(W)

    V = n_var(H)

    V_1 = V[:, :, 1:]
    V_2 = V[:, :, -1:]

    V = T.concatenate([V_1, V_2], axis=2)
    Z = T.concatenate([F, V], axis=0)

    
    repr_cost = T.mean((n_dec(n_lstm(Z)) - Y)**2)

    return repr_cost + vari_cost

trainer = AdamTrainer(rng, batchsize=BATCH_SIZE, epochs=3000, alpha=0.0005, cost=cost)
trainer.train(network, M_I, E, M_O, filename=[[None, '../models/vae_lstm/3_vae_lstm_layer_0.npz', None, None,
                                        None, '../models/vae_lstm/3_vae_lstm_layer_1.npz', None, None,],
                                        [None,],
                                        [None, None, '../models/vae_lstm/3_vae_lstm_layer_2.npz', None,
                                        None, None, '../models/vae_lstm/3_vae_lstm_layer_3.npz', None,],
                                        [None, '../models/vae_lstm/3_vae_lstm_layer_4.npz', None, None,
                                        None, '../models/vae_lstm/3_vae_lstm_layer_5.npz', None, None,],
                                        ['../models/vae_lstm/3_vae_lstm_layer_6.npz',]])


