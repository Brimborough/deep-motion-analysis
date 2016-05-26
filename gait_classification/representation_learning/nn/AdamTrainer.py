import sys
import numpy as np
import timeit
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
        elif cost == 'binary_cross_entropy':
            self.y_pred = lambda network, x: network(x)
            self.error = lambda network, y_pred, y: T.mean(T.neq(T.argmax(y_pred, axis=1), y))
            self.cost   = lambda network, y_pred, y: T.nnet.binary_crossentropy(y_pred, y).mean()
        elif cost == 'cross_entropy':
            self.y_pred = lambda network, x: network(x)
            self.error = lambda network, y_pred, y: T.mean(T.neq(T.argmax(y_pred, axis=1), y))
            self.cost   = lambda network, y_pred, y: T.nnet.categorical_crossentropy(y_pred, y).mean()
        else:
            self.cost = cost
        
    def regularization(self, network, target=0.0):
        return sum([T.mean(abs(p - target)) for p in network.params]) / len(network.params)
        
    def get_cost_updates(self, network, input, output):
        
        y_pred = self.y_pred(network, input)
        cost = self.cost(network, y_pred, output) + self.gamma * self.regularization(network)
        error = None

        if (self.error):
            # Only meaningful in classification
            error = self.error(network, y_pred, output)
        
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

        return (cost, updates, error)
        
    def train(self, network, train_input, train_output,
                             valid_input=None, valid_output=None,
                             test_input=None, test_output=None, filename=None):
        
        input = train_input.type()
        output = train_output.type()
        # Match batch index
        index = T.lscalar()
        
        self.params = network.params
        self.m0params = [theano.shared(np.zeros(p.shape.eval(), dtype=theano.config.floatX), borrow=True) for p in self.params]
        self.m1params = [theano.shared(np.zeros(p.shape.eval(), dtype=theano.config.floatX), borrow=True) for p in self.params]
        self.t = theano.shared(np.array([1], dtype=theano.config.floatX))
        
        cost, updates, error = self.get_cost_updates(network, input, output)

        train_func = theano.function(inputs=[index], 
                                     outputs=[cost, error], 
                                     updates=updates, 
                                     givens={input:train_input[index*self.batchsize:(index+1)*self.batchsize],
                                             output:train_output[index*self.batchsize:(index+1)*self.batchsize],}, 
                                     allow_input_downcast=True)

        valid_func = None
        if (valid_input):
            # Full batch evaluation
            valid_batchsize=valid_input.get_value().shape[0]

            valid_func = theano.function(inputs=[index],
                                         outputs=[cost, error],
                                         givens={input:valid_input[index*valid_batchsize:(index+1)*valid_batchsize],
                                                 output:valid_output[index*valid_batchsize:(index+1)*valid_batchsize],},
                                         allow_input_downcast=True)

        test_func = None
        if (test_input):
            # Full batch evaluation
            test_batchsize=test_input.get_value().shape[0]

            test_func = theano.function(inputs=[index],
                                        outputs=[cost, error],
                                        givens={input:test_input[index*test_batchsize:(index+1)*test_batchsize],
                                                output:test_output[index*test_batchsize:(index+1)*test_batchsize],},
                                        allow_input_downcast=True)

        ###############
        # TRAIN MODEL #
        ###############
        print('... training')
        
        best_valid_error = 1.
        best_epoch = 0

        last_tr_mean = 0.

        start_time = timeit.default_timer()

        for epoch in range(self.epochs):
            
            batchinds = np.arange(train_input.shape.eval()[0] // self.batchsize)
            self.rng.shuffle(batchinds)
            
            sys.stdout.write('\n')
            
            tr_costs  = []
            tr_errors = []
            for bii, bi in enumerate(batchinds):
                tr_cost, tr_error = train_func(bi)
                tr_costs.append(tr_cost)
                tr_errors.append(tr_error)
                if np.isnan(tr_costs[-1]): return
                if bii % (int(len(batchinds) / 1000) + 1) == 0:
                    sys.stdout.write('\r[Epoch %i]  %0.1f%% mean training error: %.5f' % (epoch, 100 * float(bii)/len(batchinds), np.mean(tr_errors) * 100.))
                    sys.stdout.flush()

            curr_tr_mean = np.mean(tr_errors)
            diff_tr_mean, last_tr_mean = curr_tr_mean-last_tr_mean, curr_tr_mean

            # TODO: is there any need to provide an index at all?
            valid_cost, valid_error = valid_func(0)
            valid_diff = valid_error - best_valid_error

            sys.stdout.write('\r[Epoch %i] 100.0%% mean training error: %.5f training diff: %.5f validation error: %.5f validation diff: %.5f %s\n' % 
                (epoch, curr_tr_mean * 100., diff_tr_mean * 100., valid_error * 100., valid_diff * 100., str(datetime.now())[11:19]))
            sys.stdout.flush()

            # if we got the best validation score until now
            if valid_error < best_valid_error:
                best_valid_error = valid_error
                best_epoch = epoch

                # Only save the model if the validation error improved
                # TODO: Don't add time needed to save model to training time
                network.save(filename)

        end_time = timeit.default_timer()

        ####################
        # Final Validation #
        ####################

        # TODO: is there any need to provide an index at all?
        test_cost, test_error = test_func(0)

        sys.stdout.write(('Optimization complete. Best validation score of %f %% '
                          'obtained at epoch %i, with test performance %f %%\n') %
                         (best_valid_error * 100., best_epoch + 1, test_error * 100.))
        sys.stdout.flush()

        sys.stdout.write(('Training took %.2fm\n' % ((end_time - start_time) / 60.)))
        sys.stdout.flush()
