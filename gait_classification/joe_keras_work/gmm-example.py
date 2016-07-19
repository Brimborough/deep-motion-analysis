from keras.models import Sequential
from keras.layers.core import Dense
from keras import backend as back
# TODO: This is only implemented for theano, rewrite it using keras.backend (as an exercise).
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1067)


def softmax(x):
    # softmaxes the columns of x
    # z = x - np.max(x, axis=0, keepdims=True) # for safety
    e = np.exp(x)
    en = e / np.sum(e, axis=1, keepdims=True)
    return en


'''
pi : n,m m components for mixture model
mu:  n,m*d for each comp, there are d mean for d features
sigma: n,m*d for each comp, there are should d*d cov for d features, but we only consider the diag here
return samples from mixture guassian model
'''


def draw_sample_from_mixture_guassian(n, d, out_pi, out_mu, out_sigma):
    result = np.zeros((n, d))
    m = out_pi.shape[1]
    for i in range(n):
        c = int(np.random.choice(range(m), size=1, replace=True, p=out_pi[i, :]))
        mu = out_mu[i, c * d:(c + 1) * d].ravel()
        sig = np.diag(out_sigma[i, c * d:(c + 1) * d])
        sample_c = np.random.multivariate_normal(mu, sig ** 2, 1).ravel()
        result[i, :] = sample_c
    return result


def drawContour(m, X, Y):
    n = 50
    xx = np.linspace(0, 1, n)
    yy = np.linspace(0, 1, n)
    xm, ym = np.meshgrid(xx, yy)
    logps = np.zeros((xm.size, 1))
    xm1 = xm.reshape(xm.size, 1)
    ym1 = ym.reshape(ym.size, 1)
    for i in range(xm.size):
        logps[i] = m.evaluate(xm1[i], ym1[i])
    plt.figure(figsize=(10, 10))
    plt.scatter(X, Y, color='g')
    plt.contour(xm, ym, np.reshape(logps, (n, n)), levels=np.linspace(logps.min(), logps.max(), 20))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('3-component Gaussian Mixture Model for P(y|x)')


def drawSample(model, X, Y, M, title_str=''):
    y_pred = np.zeros_like(Y)
    param_pred = model.predict(X)
    if len(X.shape) > 1:
        D = X.shape[1]
    else:
        D = 1
    al = softmax(param_pred[:, 2 * D * M:])
    mu = param_pred[:, :D * M]
    sig = np.exp(param_pred[:, D * M: 2 * D * M])
    y_pred = draw_sample_from_mixture_guassian(y_pred.size, D, al, mu, sig)
    rmse = np.sqrt(((y_pred - Y) ** 2).mean())
    print("rmse : " + str(rmse))
    plt.figure(figsize=(10, 10))
    plt.plot(X, y_pred, 'b+')
    plt.plot(X, Y, 'g-.')
    plt.legend(['pred', 'truth'])
    plt.title(title_str)

def np_log_sum_exp(x, axis=None):
    x_max = np.max(x, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True,dtype=np.float32)) + x_max
'''
avoid nan when comput log_sum_exp in mdn loss function
https://amjadmahayri.wordpress.com/2014/04/30/mixture-density-networks/
https://github.com/Theano/Theano/issues/1563
'''
def log_sum_exp(x, axis=None):
    x_max = T.max(x, axis=axis, keepdims=True)
    return T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max

def np_loss(true, parameters):
    two_pi = 2 * np.pi
    D = true.shape[1]
    M = parameters.shape[1] / (2 * D + 1)

    mu = parameters[:, 0: D * M]
    sig = np.exp(parameters[:, D * M:2 * D * M])
    al = softmax(parameters[:, 2 * D * M:])

    n, k = mu.shape  # number of mixture components
    z = np.zeros((n, M))
    for c in range(M):
        z[:,c:c+1] = np.sum(((true - mu[:, c * D:(c + 1) * D]) / sig[:, c * D:(c + 1) * D]) ** 2, axis=1, keepdims=True, dtype=np.float32)/ -2.0
        normalizer = np.prod(sig[:, c * D:(c + 1) * D], axis=1, keepdims=True, dtype=np.float32) * two_pi
        z[:,c:c+1] = z[:,c:c+1] + np.log(al[:, c:c + 1]) - np.log(normalizer)
    lp = np_log_sum_exp(z,1)

    # print lp
    loss = -np.sum(lp) / n
    return loss


def neg_log_normal_mixture_likelihood(true, parameters):
    D = T.shape(true)[1]
    M = T.shape(parameters)[1] / (2 * D + 1)

    means = parameters[:, : D * M]
    sigmas = T.exp(parameters[:, D * M: 2 * D * M])
    weights = T.nnet.softmax(parameters[:, 2 * D * M:])
    two_pi = 2 * np.pi

    def component_normal_likelihood(i, mus, sis, als, tr):
        mu = mus[:, i * D:(i + 1) * D]
        sig = sis[:, i * D:(i + 1) * D]
        al = als[:, i]

        z = T.sum(((true - mu) / sig) ** 2, axis=1)/ -2.0
        normalizer = (T.prod(sig, axis=1) * two_pi)
        z += T.log(al) - T.log(normalizer)
        return z

    r, _ = theano.scan(
            fn=component_normal_likelihood,
            outputs_info=None,
            sequences=T.arange(M),
            non_sequences=[means, sigmas, weights, true])
    lp = log_sum_exp(r,0)
    loss = -T.sum(lp) / T.shape(true)[0]
    return loss



# generate some 1D regression data (reproducing Bishop book data, page 273).
# Note that the P(y|x) is not a nice distribution. E.g. it has three modes for x ~= 0.5
N = 400
X = np.linspace(0, 1, N)
Y = X + 0.3 * np.sin(2 * 3.1415926 * X) + np.random.uniform(-0.05, 0.05, N)
X, Y = Y, X

# build nerual network
M = 3
input_output_size = 256
hidden_size = 30
model_gmm = Sequential()
model_gmm.add(Dense(hidden_size, input_dim=input_output_size))
model_gmm.add(Dense((2 * input_output_size + 1) * M))
model_gmm.compile(loss=neg_log_normal_mixture_likelihood, optimizer='rmsprop')

get_3rd_layer_output = back.function([model_gmm.layers[0].input],
                                     [model_gmm.layers[1].output])
print(X.reshape(N,input_output_size))
out_param_value = get_3rd_layer_output([X.reshape(N,input_output_size)])[0]
print np_loss(Y.reshape(N,input_output_size), out_param_value)
print(Y)
print "before fit"
drawSample(model_gmm, X, Y, M, "sample before training")
drawContour(model_gmm, X, Y)
plt.show()
print "fitting....."
history = model_gmm.fit(X, Y, batch_size=N, nb_epoch=10000)
print "after fit"
drawSample(model_gmm, X, Y, M, "sample after training")
drawContour(model_gmm, X, Y)
plt.show()