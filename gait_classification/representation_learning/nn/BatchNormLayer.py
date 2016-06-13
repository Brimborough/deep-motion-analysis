import numpy as np
import theano
import theano.tensor as T

from theano.ifelse import ifelse

class BatchNormLayer(object):
    """
    This layer implements batch normalisation, a technique presented in [1]
    which helps to accelerate learning. This is done by transforming the input 
    to have 0 mean and a std. dev. of one. Note that through the trainable parameters beta (shifting)
    and gamma (scaling), the network can learn to undo this normalisation. 

    Notes
    ---------- This layer should be inserted between a linear transformation (e.g. Conv2DLayer or HiddenLayer) and the
    respective ActivationLayer.
    ----------
    References
    ----------
       [1] Ioffe, Sergey and Szegedy, Christian (2015):
           Batch Normalization: Accelerating Deep Network Training by Reducing
           Internal Covariate Shift. http://arxiv.org/abs/1502.03167.
    """

    def __init__(self, rng, shape, axes=(0,), epsilon=1e-10):
        self.axes = axes
        self.shape = [(1 if si in axes else s) for si,s in enumerate(shape)]
        self.beta = theano.shared(value = np.zeros(self.shape, dtype=theano.config.floatX), name='beta')
        self.gamma = theano.shared(value = np.ones(self.shape, dtype=theano.config.floatX), name='gamma')
        self.epsilon = epsilon
        self.params = [self.beta, self.gamma]
        
    def __call__(self, input): 
        mean = input.mean(self.axes, keepdims=True) 
        std = input.std(self.axes, keepdims=True) + self.epsilon 

        # Don't batch normalise a single data point
        mean = ifelse(T.gt(input.shape[0], 1), mean, T.zeros(mean.shape, dtype=mean.dtype))
        std  = ifelse(T.gt(input.shape[0], 1), std, T.ones(std.shape, dtype=std.dtype))

        return (input - mean) * T.addbroadcast((self.gamma / std) + self.beta, *self.axes)

    def inv(self, output): 
        return output

    '''
        Save extra parameters for loading of the layer within other layers, i.e. the LSTM
    '''
    def load(self, filename):
        if filename is None: return
        if not filename.endswith('.npz'): filename += '.npz'
        data = np.load(filename)
        self.gamma = theano.shared(value=data['gamma'].astype(theano.config.floatX), name='gamma')
        self.beta = theano.shared(value=data['beta'].astype(theano.config.floatX), name='beta')
        self.params = [self.gamma, self.beta]

    def save(self, filename):
        if filename is None: return
        np.savez_compressed(filename,
                            gamma=np.array(self.gamma.eval()),
                            beta=np.array(self.beta.eval()))


class InverseBatchNormLayer(BatchNormLayer):
    """Inverse of the BatchNormLayer"""
    def __init__(self, shape):
        super(InverseBatchNormLayer, self).__init__(shape=shape)
        self.inv, self.__call__ = self.__call__, self.inv
