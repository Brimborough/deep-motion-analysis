import numpy as np
import theano
import theano.tensor as T

# Seperate labeled from unlabeled data
x = np.array([[0.05, 0.05, 0.9],
              [0.4, 0.2, 0.4],
              [0.3, 0.2, 0.5],
              [0.8, 0.1, 0.1]])

y = np.array([[0., 0., 0.],
              [1., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.]])

X = T.dmatrix()
Y = T.dmatrix()

label   = lambda X, Y: X[T.nonzero(Y)[0]]
unlabel = lambda X, Y: X[T.nonzero(1.-T.sum(Y, axis=1))]
split   = lambda X, Y: [label(X, Y), unlabel(X,Y)]

# Predictions
pred    = lambda Y: T.argmax(Y, axis=1)

label_func   = theano.function([X, Y], label(X, Y))
unlabel_func = theano.function([X, Y], unlabel(X, Y))
split_func   = theano.function([X, Y], split(X,Y))

# Make sure error is only calculated for labeled data
#error_func = theano.function([X, Y], T.mean(T.neq(T.argmax(label(X, Y), axis=1), T.argmax(label(Y, Y), axis=1))))
error_func = theano.function([X, Y], T.mean(T.neq(pred(label(X, Y)), pred(label(Y, Y)))))

l, u = split_func(x,y)

print 'labeled:'
print l
print 'unlabeled:'
print u
print 'error:'
print error_func(x, y)
