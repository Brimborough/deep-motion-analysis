import numpy as np
import theano
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams

class HiddenLayer(object):

    def __init__(self, rng, weights_shape, W=None, b=None):
        
        self.theano_rng = RandomStreams(rng.randint(2 ** 30))
        
        if W is None:
            # TODO: W_bound depends on the activation function
            W_bound = np.sqrt(6. / np.prod(weights_shape))
            W = np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=weights_shape),
                dtype=theano.config.floatX)
        
        if b is None:
            b = np.zeros((weights_shape[1],), dtype=theano.config.floatX)

        self.input_units = W[0]
        self.output_units = W[1]
        
        self.W = theano.shared(value=W, borrow=True)
        self.b = theano.shared(value=b, borrow=True)
        
        self.params = [self.W, self.b]
        
    def __call__(self, input):
        return input.dot(self.W) + self.b
        
    def inv(self, output):
        return self.W.dot((output - self.b).T).T 
        
    def load(self, filename):
        if filename is None: return
        if not filename.endswith('.npz'): filename+='.npz'
        data = np.load(filename)
        self.W = theano.shared(data['W'].astype(theano.config.floatX), borrow=True)
        self.b = theano.shared(data['b'].astype(theano.config.floatX), borrow=True)
    
    def save(self, filename):
        if filename is None: return
        np.savez_compressed(filename,
            W=np.array(self.W.eval()),
            b=np.array(self.b.eval()))
        
        
