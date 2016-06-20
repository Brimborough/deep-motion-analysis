import numpy as np
import theano
import pprint
from theano.compat.python2x import OrderedDict
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

# Test how outcome of vector dot matrix
