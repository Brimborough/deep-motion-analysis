import numpy as np
import theano
import theano.tensor as T

from nn.ActivationLayer import ActivationLayer
from nn.AdamTrainer import AdamTrainer
from nn.Conv1DLayer import Conv1DLayer
from nn.HiddenLayer import HiddenLayer 
from nn.Network import Network, AutoEncodingNetwork
from nn.NoiseLayer import NoiseLayer
from nn.Pool1DLayer import Pool1DLayer
from nn.ReshapeLayer import ReshapeLayer

from tools.utils import load_cmu, load_cmu_small

from nn.AnimationPlotLines import animation_plot

rng = np.random.RandomState(23455)

shared = lambda d: theano.shared(d, borrow=True)

dataset, std, mean = load_cmu_small(rng)

#(17000, 66, 240)
train_set_x_first = shared(dataset[0][0][:, :, :160])
train_set_x_second = shared(dataset[0][0][:, :, 180:])

batchsize = 1

network_predict = AutoEncodingNetwork(Network(
    Conv1DLayer(rng, (64, 66, 25), (batchsize, 66, 160)),
    ActivationLayer(rng, f='ReLU'),
    Pool1DLayer(rng, (2,), (batchsize, 64, 160)),

    Conv1DLayer(rng, (128, 64, 25), (batchsize, 64, 80)),
    ActivationLayer(rng, f='ReLU'),
    Pool1DLayer(rng, (2,), (batchsize, 128, 80)),
    
    Conv1DLayer(rng, (256, 128, 25), (batchsize, 128, 40)),
    ActivationLayer(rng, f='ReLU'),
    Pool1DLayer(rng, (2,), (batchsize, 256, 40)),
))

network_predict.load(['../models/cmu/predict_v_1_overlap/layer_0.npz', None, None,   # 1. Conv, Activation, Pooling
             '../models/cmu/predict_v_1_overlap/layer_1.npz', None, None,   # 2. Conv, Activation, Pooling
             '../models/cmu/predict_v_1_overlap/layer_2.npz', None, None,]) # 3. Conv, Activation, Pooling

network_denoise = AutoEncodingNetwork(Network(
    Conv1DLayer(rng, (64, 66, 25), (batchsize, 66, 240)),
    ActivationLayer(rng, f='ReLU'),
    Pool1DLayer(rng, (2,), (batchsize, 64, 240)),

    Conv1DLayer(rng, (128, 64, 25), (batchsize, 64, 120)),
    ActivationLayer(rng, f='ReLU'),
    Pool1DLayer(rng, (2,), (batchsize, 128, 120)),
    
    Conv1DLayer(rng, (256, 128, 25), (batchsize, 128, 60)),
    ActivationLayer(rng, f='ReLU'),
    Pool1DLayer(rng, (2,), (batchsize, 256, 60)),
))

network_denoise.load(['../models/cmu/conv_ae/layer_0.npz', None, None,   # 1. Conv, Activation, Pooling
             '../models/cmu/conv_ae/layer_1.npz', None, None,   # 2. Conv, Activation, Pooling
             '../models/cmu/conv_ae/layer_2.npz', None, None,]) # 3. Conv, Activation, Pooling

trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=50, alpha=0.00001, l1_weight=0.1, l2_weight=0.0, cost='mse')

result_noisy = trainer.get_representation(network_predict, train_set_x_first, len(network_predict.network.layers) - 1)

result_noisy = np.concatenate([dataset[0][0][:, :, :160], result_noisy[:, :, 80:]], axis=2)
result_noisy_shared = shared(result_noisy)

result = trainer.get_representation(network_denoise, result_noisy_shared, len(network_denoise.network.layers) - 1)  * (std + 1e-10) + mean

dataset_ = dataset[0][0] * (std + 1e-10) + mean

# Nice walking 50

new1 = result[70:71]
new2 = dataset_[70:71]

animation_plot([new1, new2], interval=15.15)

#trainer.train(network=network, train_input=train_set_x_first, train_output=train_set_x_second,
#              filename=[None, '../models/cmu/predict_v_0/layer_0.npz', None, None,   # Noise, 1. Conv, Activation, Pooling
#                              '../models/cmu/predict_v_0/layer_1.npz', None, None,   # 2. Conv, Activation, Pooling
#                              '../models/cmu/predict_v_0/layer_2.npz', None, None,]) # 3. Conv, Activation, Pooling
	
