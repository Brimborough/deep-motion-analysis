import numpy as np
import theano 
import theano.tensor as T

x = np.zeros([100, 1, 28, 28])
#a = np.zeros([4, 28, 28, 10])
a = np.zeros([1, 28, 28, 10])

t = T.dtensor4()
o = T.iscalar()

s = t[:,:,:,o].dimshuffle('x', 0, 1, 2)
shuff_f = theano.function([t, o], s)

mu = shuff_f(a,0) * (x*shuff_f(a,1) + shuff_f(a,2)) + x*shuff_f(a,3) + shuff_f(a,4)
v = shuff_f(a,0) * (x*shuff_f(a,1) + shuff_f(a,2)) + x*shuff_f(a,3) + shuff_f(a,4)

print ((x-mu)*v+mu).shape
