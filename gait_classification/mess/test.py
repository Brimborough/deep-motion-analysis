import numpy as np
import theano
import pprint
from theano.compat.python2x import OrderedDict
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import sys

sys.path.append('../representation_learning/nn')

from BatchNormLayer import BatchNormLayer


"""

    Write test for truncated gradients only working for few time steps
    Write test for clipped gradients, testing.

"""

# Write simple scan, truncating gradients for x ** 2 to 2 on a sequence of 4, check updates correspond with what you believe

