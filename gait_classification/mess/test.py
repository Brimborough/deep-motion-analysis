import numpy as np
import theano
import pprint
from theano.compat.python2x import OrderedDict
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import sys

sys.path.append('../representation_learning/nn')

from BatchNormLayer import BatchNormLayer


class bn(object):

    def createBN(self, bn, batch_size, input_shape, hidden_shape):
        data = OrderedDict()
        for i in ['input','hidden','cell']:
            axes = False
            epsilon = False

            if i == 'input':
                shape = input_shape
            else:
                shape = hidden_shape

            if i+'_axes' in bn:
                axes =  bn[i+'_axes']
            if i + '_epsilon' in bn:
                epsilon =  bn[i + '_epsilon']

            if axes and epsilon:
                print epsilon
                data[i] = BatchNormLayer(None, [batch_size, shape], axes=axes, epsilon=epsilon)
            elif axes:
                print axes
                data[i] = BatchNormLayer(None, [batch_size, shape], axes=axes)
            elif epsilon:
                data[i] = BatchNormLayer(None, [batch_size, shape], epsilon=epsilon)
            else:
                data[i] = BatchNormLayer(None, [batch_size, shape])

        return data.values()

b = bn()

c,d,n = b.createBN({'input_axes':(2,)},2,2,2)

print c

print d.epsilon