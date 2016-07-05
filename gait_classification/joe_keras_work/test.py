import numpy as np
import theano
import pprint
from theano.compat.python2x import OrderedDict
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import sys

sys.path.append('../representation_learning/nn')

from LSTM import LSTM


"""

        SHAPE FOR LSTM: TimeSeries, Batch, Feature_size
        input.dimshuffle(1,0,2)

16*90 90*90 = 16*90
"""

#Add IMDB DATA here

