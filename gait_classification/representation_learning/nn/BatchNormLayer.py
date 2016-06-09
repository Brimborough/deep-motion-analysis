import numpy as np
import theano
import theano.tensor as T

from theano.tensor.nnet.bn import batch_normalization

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

    def __init__(self, shape, mode='low_mem'):
        # BN shift parameter, must be of same dimensionality as inputs and broadcastable against it
        # One shift parameter for each dimension
        self.beta = theano.shared(value = np.zeros(shape[1:], dtype=theano.config.floatX), name='beta')
        # BN scale parameter, must be of same dimensionality as inputs and broadcastable against it
        # One scale paramter for each dimension
        self.gamma = theano.shared(value = np.ones(shape[1:], dtype=theano.config.floatX), name='gamma')

        # 'low_mem' or 'high_mem'
        self.mode = mode

        self.params = [self.beta, self.gamma]
        
    def __call__(self, input): 
        bn = batch_normalization(input, self.gamma, self.beta, 
                                 input.mean((0,), keepdims=True), 
                                 input.std((0,), keepdims=True), 
                                 self.mode)

        # In case a batchnormalisation of scalars is attempted,
        # the result might be a tensor of nans
        if (T.isnan(bn.flatten(ndim=1)[0])): 
            return input

        return bn

    def inv(self, output): 
        return output
        
    def load(self, filename): pass
    def save(self, filename): pass

class InverseBatchNormLayer(BatchNormLayer):
    """Inverse of the BatchNormLayer"""
    def __init__(self, shape, mode='low_mem'):
        super(InverseBatchNormLayer, self).__init__(shape=shape, mode=mode)
        self.inv, self.__call__ = self.__call__, self.inv
