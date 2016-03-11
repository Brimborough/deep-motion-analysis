import sys
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from datetime import datetime

class AdamTrainer:
    
    def __init__(self, rng, batchsize, epochs=100, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08, gamma=0.1, cost='mse'):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.gamma = gamma
        self.rng = rng
        self.theano_rng = RandomStreams(rng.randint(2 ** 30))
        self.epochs = epochs
        self.batchsize = batchsize
        if   cost == 'mse':
            self.cost = lambda network, x, y: T.mean((network(x) - y)**2)
        elif cost == 'cross_entropy':
            self.cost = lambda network, x, y: T.nnet.binary_crossentropy(network(x), y).mean()
        else:
            self.cost = cost
        
    def regularization(self, network, target=0.0):
        return sum([T.mean(abs(p - target)) for p in network.params]) / len(network.params)
        
    def get_cost_updates(self, network, input, output):
        
        cost = (self.cost(network, input, output) + self.gamma * self.regularization(network))
        
        gparams = T.grad(cost, self.params)
        m0params = [self.beta1 * m0p + (1-self.beta1) *  gp     for m0p, gp in zip(self.m0params, gparams)]
        m1params = [self.beta2 * m1p + (1-self.beta2) * (gp*gp) for m1p, gp in zip(self.m1params, gparams)]
        params = [p - (self.alpha / self.batchsize) * 
                  ((m0p/(1-(self.beta1**self.t[0]))) /
            (T.sqrt(m1p/(1-(self.beta2**self.t[0]))) + self.eps))
            for p, m0p, m1p in zip(self.params, m0params, m1params)]
        
        updates = ([( p,  pn) for  p,  pn in zip(self.params, params)] +
                   [(m0, m0n) for m0, m0n in zip(self.m0params, m0params)] +
                   [(m1, m1n) for m1, m1n in zip(self.m1params, m1params)] +
                   [(self.t, self.t+1)])

        return (cost, updates)
        
    def train(self, network, input_data, output_data, filename=None):
        
        input = input_data.type()
        output = output_data.type()
        # Match batch index
        index = T.lscalar()
        
        self.params = network.params
        self.m0params = [theano.shared(np.zeros(p.shape.eval(), dtype=theano.config.floatX), borrow=True) for p in self.params]
        self.m1params = [theano.shared(np.zeros(p.shape.eval(), dtype=theano.config.floatX), borrow=True) for p in self.params]
        self.t = theano.shared(np.array([1], dtype=theano.config.floatX))
        
        cost, updates = self.get_cost_updates(network, input, output)

        train_func = theano.function(inputs=[index],
                                     inputs=cost, 
                                     outputs=updates, 
                                     givens={input:input_data[index*self.batchsize:(index+1)*self.batchsize],
                                     output:output_data[index*self.batchsize:(index+1)*self.batchsize],}, 
                                     allow_input_downcast=True)
        
        last_mean = 0
        for epoch in range(self.epochs):
            
            batchinds = np.arange(input_data.shape.eval()[0] // self.batchsize)
            self.rng.shuffle(batchinds)
            
            sys.stdout.write('\n')
            
            c = []
            for bii, bi in enumerate(batchinds):
                c.append(train_func(bi))
                if np.isnan(c[-1]): return
                if bii % (int(len(batchinds) / 1000) + 1) == 0:
                    sys.stdout.write('\r[Epoch %i]  %0.1f%% mean %.5f' % (epoch, 100 * float(bii)/len(batchinds), np.mean(c)))
                    sys.stdout.flush()

            curr_mean = np.mean(c)
            diff_mean, last_mean = curr_mean-last_mean, curr_mean
            sys.stdout.write('\r[Epoch %i] 100.0%% mean %.5f diff %.5f %s' % 
                (epoch, curr_mean, diff_mean, str(datetime.now())[11:19]))
            sys.stdout.flush()
            
            network.save(filename)
