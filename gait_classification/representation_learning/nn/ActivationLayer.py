import numpy as np
import theano
import theano.tensor as T

class ActivationLayer(object):

    def __init__(self, rng, f='ReLU', g=lambda x: x, params=None):
        if   f == 'ReLU':
            if hasattr(T.nnet, 'relu'):
                self.f = T.nnet.relu
            else:
                self.f = lambda x: T.switch(x<0,0,x)
            self.g = lambda x: x
        elif f == 'PReLU':
            # Avoids dying ReLU units
            if hasattr(T.nnet, 'relu'):
                self.f = lambda x: T.nnet.relu(x, alpha=0.01)
            else:
                self.f = lambda x: T.switch(x<=0,a*x,x)
            self.g = lambda x: x
        elif f == 'elu':
            self.f = T.nnet.elu
            self.g = lambda x: x
        elif f == 'tanh':
            self.f = T.tanh
            self.g = T.arctanh
        elif f == 'sigmoid':
            self.f = T.nnet.sigmoid
            self.g = lambda x: x
        elif f == 'softmax':
            self.f = T.nnet.softmax
            self.g = lambda x: x
        elif f == 'softplus':
            self.f = T.nnet.softplus
            self.g = lambda x: x
        elif f == 'identity':
            self.f = lambda x: x
            self.g = lambda x: x
        else:
            self.f = f
            self.g = g
        
        self.params = [] if params is None else params
        
    def __call__(self, input): 
        return self.f(input)

    def inv(self, output): 
        return self.g(output)
        
    def load(self, filename): pass
    def save(self, filename): pass
    def reset(self): pass
        
