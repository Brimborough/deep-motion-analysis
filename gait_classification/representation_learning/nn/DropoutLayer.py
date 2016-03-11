import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class DropoutLayer(object):

    def __init__(self, rng, amount=0.3):
        self.amount = amount
        self.theano_rng = RandomStreams(rng.randint(2 ** 30))
        self.params = []
        
    def __call__(self, input):
        if self.amount > 0.0:
            return (input * self.theano_rng.binomial(
                size=input.shape, n=1, p=(1-self.amount),
                dtype=theano.config.floatX)) / (1-self.amount)
        else:
            return input
        
    def inv(self, output):
        return output
        
    def load(self, filename): pass
    def save(self, filename): pass
        