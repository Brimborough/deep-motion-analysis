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
from nn.LSTM1DLayerTest import LSTM1DLayer
from nn.Network import Network, AutoEncodingNetwork, InverseNetwork
from nn.AnimateFrames import plot_movement

from tools.utils import load_locomotion
from tools.utils import load_terrain

rng = np.random.RandomState(23455)

shared = lambda d: theano.shared(d, borrow=True)
#dataset, std, mean = load_locomotion(rng)
dataset, std, mean = load_terrain(rng)

#print dataset[0][0][1500:1520].shape

#train_control_dataset = dataset[0][0][300:320]
train_control_dataset = dataset[0][0][1500:1520]
train_control_dataset = train_control_dataset.swapaxes(0, 1)[-3:]
train_control_dataset = train_control_dataset.swapaxes(0, 1)

print "control dataset shape = ", train_control_dataset.shape

C = shared(train_control_dataset)

#dataset, std, mean = load_locomotion(rng)
dataset, std, mean = load_terrain(rng)

#train_motion_input_dataset = dataset[0][0][300:320]
train_motion_input_dataset = dataset[0][0][1500:1520]
train_motion_input_dataset = train_motion_input_dataset.swapaxes(0, 1)[:-3]
train_motion_input_dataset = train_motion_input_dataset.swapaxes(0, 1)

print "motion input dataset shape = ", train_motion_input_dataset.shape

M_I = shared(train_motion_input_dataset)

BATCH_SIZE = 20
H_SIZE = 128

encoder = HiddenLayer(rng, (128, H_SIZE))
encode_igate = HiddenLayer(rng, (128, H_SIZE))
encode_fgate = HiddenLayer(rng, (128, H_SIZE))

recoder = HiddenLayer(rng, (H_SIZE, H_SIZE))
recode_igate = HiddenLayer(rng, (H_SIZE, H_SIZE))
recode_fgate = HiddenLayer(rng, (H_SIZE, H_SIZE))

activation = ActivationLayer(rng, f='elu')
dropout = DropoutLayer(rng, 0.2)

encoder_network = Network(
    Conv1DLayer(rng, (64, 63, 25), (BATCH_SIZE, 63, 240)),
    Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 240)),
    ActivationLayer(rng, f='elu'),

    Conv1DLayer(rng, (256, 64, 25), (BATCH_SIZE, 64, 120)),
    Pool1DLayer(rng, (2,), (BATCH_SIZE, 256, 120)),
    ActivationLayer(rng, f='elu'),
)

encoder_network.load(['../models/vae_lstm/3_vae_lstm_layer_0.npz', None, None,
                        '../models/vae_lstm/3_vae_lstm_layer_1.npz', None, None,])

ff_network = Network(
    Conv1DLayer(rng, (64, 3, 25), (BATCH_SIZE, 3, 240)),
    Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 240)),
    ActivationLayer(rng, f='elu'),

    Conv1DLayer(rng, (128, 64, 25), (BATCH_SIZE, 64, 120)),
    Pool1DLayer(rng, (2,), (BATCH_SIZE, 128, 120)),
    ActivationLayer(rng, f='elu'),
)

ff_network.load(['../models/vae_lstm/3_vae_lstm_layer_4.npz', None, None,
                   '../models/vae_lstm/3_vae_lstm_layer_5.npz', None, None,])


lstm_network = Network(
	LSTM1DLayer(encoder, recoder, encode_igate, recode_igate, encode_fgate, recode_fgate, activation, dropout, dropout),
)

lstm_network.load(['../models/vae_lstm/3_vae_lstm_layer_6.npz'])

var_network = Network(
    VariationalLayer(rng, sample=False),
)

decoder_network = Network(
    InverseNetwork(Pool1DLayer(rng, (2,), (BATCH_SIZE, 128, 120))),
    Conv1DLayer(rng, (128, 128, 25), (BATCH_SIZE, 128, 120)),
    ActivationLayer(rng, f='elu'),

    InverseNetwork(Pool1DLayer(rng, (2,), (BATCH_SIZE, 128, 240))),  
    Conv1DLayer(rng, (66, 128, 25), (BATCH_SIZE, 128, 240)),
    ActivationLayer(rng, f='elu'),
)

decoder_network.load([None, '../models/vae_lstm/3_vae_lstm_layer_2.npz', None,
                        None, '../models/vae_lstm/3_vae_lstm_layer_3.npz', None,])

enc_func = theano.function([], encoder_network(M_I))
enc_result = enc_func()
print "encoder result shape = ", enc_result.shape

H = shared(enc_result)

#print "generate samples..."
#H = shared(np.random.normal(0, 1, enc_result.shape))
#H = shared(np.random.normal(200, 100, (2, 256, 60)))

ff_func = theano.function([], ff_network(C))
ff_result = ff_func()
print "feed-forward result shape = ", ff_result.shape
L = shared(ff_result)

var_func = theano.function([], var_network(H))
var_result = var_func()
print "var result shape = ", var_result.shape
H = shared(var_result)

H = H[:, :, 1:]
H_last = H[:, :, -1:]

H = T.concatenate([H, H_last], axis=2)
Z = T.concatenate([L, H], axis=0)

#print "concatenate shape = ", Z.eval().shape

lstm_func = theano.function([], lstm_network(Z))
lstm_result = lstm_func()
print "lstm result shape = ", lstm_result.shape
E = shared(lstm_result)

dec_func = theano.function([], decoder_network(E))
dec_result = dec_func()
print "decoder result shape = ", dec_result.shape

result = dec_result * std + mean

print "result shape = ", result.shape

np.savez_compressed("vae_lstm_results_test_data.npz", clips=result)

clip = 10
new1 = result[clip:clip+1]
new2 = result[clip+1:clip+2]

animation_plot([new1, new2], filename='vae_lstm_clip10.mp4', interval=15.15)

