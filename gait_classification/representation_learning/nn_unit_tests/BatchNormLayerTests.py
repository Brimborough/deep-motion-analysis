import numpy as np
import theano
import theano.tensor as T
import sys

sys.path.append('../nn')

from BatchNormLayer import BatchNormLayer

N  = 4
n_out = 4

x = np.array(np.arange(N*n_out).reshape(N, n_out), dtype='float32')
norm_x = (x - x.mean((0,), keepdims=True)) / x.std((0,), keepdims=True)

t_x = T.fmatrix('x')
t_bn = BatchNormLayer((N, n_out,))
t_f = theano.function([t_x], t_bn(t_x))

# Test case 1: Batchnorm with shifting = 0 and scaling = 1 on 2D-Tensor
t_norm_x = t_f(x)
assert (np.sum(abs(t_norm_x - norm_x)) < 1e-5), (
    'BatchNormLayer gives incorrect result for Test case 1. '
    'Correct output is {0} but returned output is {1}.'
    .format(norm_x, t_f(x))
)

# reset beta, gamma
batch_size = 2
channels   = 2
rows       = 2
cols       = 2
t_bn2 = BatchNormLayer((batch_size, channels, rows, cols))

x2 = np.array(np.arange(batch_size*channels*rows*cols).reshape(
             batch_size, channels, rows, cols), dtype='float32')

t_x2 = T.ftensor4('x2')

t_f2 = theano.function([t_x2], t_bn2(t_x2))

norm_x = (x - x.mean((0,), keepdims=True)) / x.std((0,), keepdims=True)

## Test case 2: Batchnorm with shifting = 0 and scaling = 1 on 4D-Tensor
t_norm_x = t_f(x)
assert (np.sum(abs(t_norm_x - norm_x)) < 1e-5), (
    'BatchNormLayer gives incorrect result for Test case 2. '
    'Correct output is {0} but returned output is {1}.'
    .format(norm_x, t_f(x)))
