import numpy as np
import theano
import pprint
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


'''
x = np.array([[ 1, 2, 3],
     [ 4, 5, 6]])

y = np.array([[1, 2], [3, 4], [5,6]])

print x.shape
print y.shape
print T.dot(x,y).eval()

'''

r = np.random.RandomState(23455)
class t(object):
    def __init__(self, mask=None):
        self.mask = 'yes' if mask is None else mask

        print self.mask

t()

