import numpy as np
import theano
import theano.tensor as T

from Param import Param

class HiddenLayer(object):

    def __init__(self, rng, weights_shape, W=None, b=None):
        
        self.weights_shape = weights_shape
        self.rng = rng
        
        if W is None:
            # TODO: W_bound depends on the activation function
            self.W_bound = np.sqrt(6. / np.prod(self.weights_shape))
            W = np.asarray(
                self.rng.uniform(low=-self.W_bound, high=self.W_bound, size=self.weights_shape),
                dtype=theano.config.floatX)
        
        if b is None:
            b = np.zeros((self.weights_shape[1],), dtype=theano.config.floatX)

        # Make sure this works for both LadderNetworks
        self.input_units = (W.shape)
        self.output_units = (W.shape)
        
        self.W = theano.shared(value=W, borrow=True)
        self.b = theano.shared(value=b, borrow=True)
        
#        self.params = [Param(self.W, True), Param(self.b, False)]
        self.params = [Param(self.W, True), Param(self.b, True)]
        
    def __call__(self, input):
        return input.dot(self.W) + self.b
        
    def inv(self, output):
        return self.W.dot((output - self.b).T).T 
        
    def load(self, filename):
        if filename is None: return
        if not filename.endswith('.npz'): filename+='.npz'
        data = np.load(filename)
        self.W = theano.shared(value=W, borrow=True)
        self.b = theano.shared(value=b, borrow=True)
        self.params = [Param(self.W, True), Param(self.b, False)]
    
    def save(self, filename):
        if filename is None: return
        np.savez_compressed(filename,
            W=np.array(self.W.eval()),
            b=np.array(self.b.eval()))
        
    def reset(self):
        W = np.asarray(
            self.rng.uniform(low=-self.W_bound, high=self.W_bound, size=self.weights_shape),
            dtype=theano.config.floatX)
        
        b = np.zeros((self.weights_shape[1],), dtype=theano.config.floatX)

        self.W = theano.shared(value=W, borrow=True)
        self.b = theano.shared(value=b, borrow=True)
        
        self.params = [Param(self.W, True), Param(self.b, False)]
