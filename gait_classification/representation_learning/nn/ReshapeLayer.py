import numpy as np
import theano
import theano.tensor as T

class ReshapeLayer(object):

    def __init__(self, rng, shape, shape_inv=None):
        self.shape = shape
        self.shape_inv = shape_inv
        self.params = []
        
    def __call__(self, input): return input.reshape(self.shape)
    def inv(self, output): return output.reshape(self.shape_inv)
        
    def load(self, filename): pass
    def save(self, filename): pass
    def reset(self): pass
