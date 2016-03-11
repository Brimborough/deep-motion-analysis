import numpy as np
import theano
import theano.tensor as T

from nn.AdamTrainer import AdamTrainer
from nn.Network import AutoEncodingNetwork

from network import network

rng = np.random.RandomState(23455)

data = np.load('data_styletransfer.npz')

#(Examples, Time frames, joints)
clips = data['clips']

clips = np.swapaxes(clips, 1, 2)
X = clips[:,:-4]

#(Motion, Styles)
classes = data['classes']


# get mean and std
preprocessed = np.load('styletransfer_preprocessed.npz')
Xmean = preprocessed['Xmean']
Xmean = Xmean.reshape(1,len(Xmean),1)
Xstd  = preprocessed['Xstd']
Xstd = Xstd.reshape(1,len(Xstd),1)

Xstd[np.where(Xstd == 0)] = 1

X = (X - Xmean) / Xstd

# Randomise data
shuffled = zip(X,Y)
np.random.shuffle(shuffled)

split = int(X.shape[0] * 0.7)

X, Y = zip(*shuffled)
X_train = np.array(X)[:split]
y_train = np.array(Y)[:split]

# TODO: Cross Validation set?
X_test = np.array(X)[split:]
Y_test = np.array(Y)[split:]

network_input  = theano.shared(X_train, borrow=True)
network_output = theano.shared(Y_train, borrow=True)

from network import network
# Load the pre-trained model
network.load([None,
              '../models/layer_0.npz', None, None,
              '../models/layer_1.npz', None, None,
              '../models/layer_2.npz', None, None,])

# Add layers for classification
network.append(ActivationLayer(rng=rng, f='softmax', g=lambda x: x, params=None))

trainer = AdamTrainer(rng, batchsize=1, epochs=25, alpha=0.00001)

# Fine-tuning for classification
trainer.train(network=ClassifyingNetwork(network), input_data=network_input, 
                                                   output_data=network_output, 
                                                   filename=[None,
                                                             'layer_0.npz', None, None,
                                                             'layer_1.npz', None, None,
                                                             'layer_2.npz', None, None,
                                                             'softmax.npz', None, None,])
