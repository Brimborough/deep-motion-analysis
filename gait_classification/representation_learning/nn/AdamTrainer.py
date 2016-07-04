import sys
import numpy as np
import timeit
import theano
import theano.tensor as T

from datetime import datetime
from ConvLadderNetwork import ConvLadderNetwork
from LadderNetwork import LadderNetwork

# To split between labeld & unlabeled examples
labeled    = lambda X, Y: X[T.nonzero(Y)[0]]
unlabeled  = lambda X, Y: X[T.nonzero(1.-T.sum(Y, axis=1))]
split_data = lambda X, Y: [labeled(X, Y), unlabeled(X, Y)]
join       = lambda X, Y: T.concatenate([X, Y], axis=0)

# Classification predictions
pred       = lambda Y: T.argmax(Y, axis=1)

class AdamTrainer(object):
    
    def __init__(self, rng, batchsize, epochs=100, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08, l1_weight=0.0, l2_weight=0.1, cost='mse'):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.rng = rng
        self.epochs = epochs
        self.batchsize = batchsize

        # Where cost is always the cost which is minimised in supervised training
        # the T.nonzero term ensures that the cost is only calculated for examples with a label
        #
        # Convetion: We mark unlabelled examples with a vector of zeros in lieu of a one-hot vector
        if   cost == 'mse':
            self.y_pred = lambda network, x: network(x)
            self.error  = lambda network, x, y: T.mean((network(x) - y)**2)
            self.cost   = lambda network, x, y: T.mean((network(x) - y)**2)
        elif cost == 'binary_cross_entropy':
            self.y_pred = lambda network, x: network(x)
            self.cost   = lambda network, y_pred, y: T.nnet.binary_crossentropy(labeled(y_pred, y), labeled(y, y)).mean()
            # classification error (taking into account only training examples with labels)
            self.error  = lambda network, y_pred, y: T.mean(T.neq(pred(labeled(y_pred, y)), pred(labeled(y, y))))
        elif cost == 'cross_entropy':
                self.y_pred = lambda network, x: network(x)
                self.cost   = lambda network, y_pred, y: T.nnet.categorical_crossentropy(labeled(y_pred, y), labeled(y, y)).mean()
                # classification error (taking into account only training examples with labels)
                self.error  = lambda network, y_pred, y: T.mean(T.neq(pred(labeled(y_pred, y)), pred(labeled(y, y))))
        else:
            self.y_pred = lambda network, x: network(x)
            self.error  = lambda network, y_pred, y: T.zeros((1,))
            self.cost   = cost

    def l1_regularization(self, network, target=0.0):
        # The if term ensures we do not regularise biases
        # TODO: This will cause problems for a single output unit, fix this
        return sum([T.mean(abs(p - target)) for p in network.params])# if (len(p.shape.eval()) > 1)])

    def l2_regularization(self, network, target=0.0):
        return sum([T.mean((p - target)**2) for p in network.params])# if (len(p.shape.eval()) > 1)])
        
    def get_cost_updates(self, network, input, output):
        
        y_pred = self.y_pred(network, input)
        cost = self.cost(network, y_pred, output) + self.l1_weight * self.l1_regularization(network) + \
                                                    self.l2_weight * self.l2_regularization(network)
        error = None

        if (self.error):
            # Only meaningful in classification
            error = self.error(network, y_pred, output)
        
        gparams = T.grad(cost, self.params)
        m0params = [self.beta1 * m0p + (1-self.beta1) *  gp     for m0p, gp in zip(self.m0params, gparams)]
        m1params = [self.beta2 * m1p + (1-self.beta2) * (gp*gp) for m1p, gp in zip(self.m1params, gparams)]
        params = [p - self.alpha * 
                  ((m0p/(1-(self.beta1**self.t[0]))) /
            (T.sqrt(m1p/(1-(self.beta2**self.t[0]))) + self.eps))
            for p, m0p, m1p in zip(self.params, m0params, m1params)]
        
        updates = ([( p,  pn) for  p,  pn in zip(self.params, params)] +
                   [(m0, m0n) for m0, m0n in zip(self.m0params, m0params)] +
                   [(m1, m1n) for m1, m1n in zip(self.m1params, m1params)] +
                   [(self.t, self.t+1)])

        return (cost, updates, error)

    def get_eval_cost_error(self, network, input, output):
        
        y_pred = self.y_pred(network, input)
        cost   = self.cost(network, y_pred, output) + self.l1_weight * self.l1_regularization(network) + \
                                                      self.l2_weight * self.l2_regularization(network)
        error = None

        if (self.error):
            # Only meaningful in classification
            error = self.error(network, y_pred, output)
        
        return (cost, error)

    def get_predictions(self, network, input):

        y_pred = pred(self.y_pred(network, input)) + 1
        return y_pred

    def create_eval_func(self, network=None, eval_input=None, eval_output=None):
        if (None in [network, eval_input]):
            # Equivalent to not defining the function
            return None

        # Match batch index
        index  = T.lscalar()
        input  = eval_input.type()
        output = eval_output.type()

        cost, error = self.get_eval_cost_error(network, input, output)

        func = theano.function(inputs=[index],
                               outputs=[cost, error],
                               givens={input:eval_input[index*self.batchsize:(index+1)*self.batchsize],
                                       output:eval_output[index*self.batchsize:(index+1)*self.batchsize],},
                               allow_input_downcast=True)

        return func

    def create_train_func(self, network=None, train_input=None, train_output=None):
        if (None in [network, train_input, train_output]):
            # Equivalent to not defining the function
            return None

        # Match batch index
        index  = T.lscalar()
        input  = train_input.type()
        output = train_output.type()

        cost, updates, error = self.get_cost_updates(network, input, output)

        func = theano.function(inputs=[index], 
                               outputs=[cost, error], 
                               updates=updates, 
                               givens={input:train_input[index*self.batchsize:(index+1)*self.batchsize],
                                       output:train_output[index*self.batchsize:(index+1)*self.batchsize],}, 
                               allow_input_downcast=True)

        return func
        
    def train(self, network, train_input, train_output,
                             valid_input=None, valid_output=None,
                             test_input=None, test_output=None, filename=None, logging=True):

        """ Conventions: For training examples with labels, pass a one-hot vector, otherwise a numpy array with zero values.
        """
        
        self.params = network.params
        self.m0params = [theano.shared(np.zeros(p.shape.eval(), dtype=theano.config.floatX), borrow=True) for p in self.params]
        self.m1params = [theano.shared(np.zeros(p.shape.eval(), dtype=theano.config.floatX), borrow=True) for p in self.params]
        self.t = theano.shared(np.array([1], dtype=theano.config.floatX))


        train_func = self.create_train_func(network=network, train_input=train_input, train_output=train_output)
        valid_func = self.create_eval_func(network=network, eval_input=valid_input, eval_output=valid_output)
        test_func  = self.create_eval_func(network=network, eval_input=test_input, eval_output=test_output)

        ###############
        # TRAIN MODEL #
        ###############
        if (logging):
            sys.stdout.write('... training\n')
        
        best_epoch = 0
        best_train_error = 1.
        best_valid_error = 1.

        last_tr_mean = 0.

        start_time = timeit.default_timer()

        for epoch in range(self.epochs):
            
            train_batchinds = np.arange(train_input.shape.eval()[0] // self.batchsize)
            self.rng.shuffle(train_batchinds)
            
            if (logging):
                sys.stdout.write('\n')
            
            tr_costs  = []
            tr_errors = []
            for bii, bi in enumerate(train_batchinds):
                tr_cost, tr_error = train_func(bi)

                # tr_error might be nan for a batch without labels in semi-supervised learning
                if not np.isnan(tr_error):
                    tr_errors.append(tr_error)

                tr_costs.append(tr_cost)
                if np.isnan(tr_costs[-1]): 
                    return
                if (logging and (bii % (int(len(train_batchinds) / 1000) + 1) == 0)):
                    sys.stdout.write('\r[Epoch %i]  %0.1f%% mean training error: %.5f' % (epoch, 100 * float(bii)/len(train_batchinds), np.mean(tr_errors) * 100.))
                    sys.stdout.flush()

            curr_tr_mean = np.mean(tr_errors)
            diff_tr_mean, last_tr_mean = curr_tr_mean-last_tr_mean, curr_tr_mean

            output_str = '\r[Epoch %i] 100.0%% mean training error: %.5f training diff: %.5f' % (epoch, curr_tr_mean * 100., diff_tr_mean * 100.)

            if (valid_func):
                valid_batchinds = np.arange(valid_input.shape.eval()[0] // self.batchsize)

                vl_errors = []
                for bii, bi in enumerate(valid_batchinds):
                    vl_cost, vl_error = valid_func(bi)
                    vl_errors.append(vl_error)

                valid_error = np.mean(vl_errors)
                valid_diff = valid_error - best_valid_error

                output_str += ' validation error: %.5f validation diff: %.5f' % (valid_error * 100., valid_diff * 100.)

            output_str += ' %s\n' % (str(datetime.now())[11:19])

            if (logging):
                sys.stdout.write(output_str)
                sys.stdout.flush()

            # Early stopping
            if (valid_func and (valid_error < best_valid_error)):
                best_valid_error = valid_error
                r_val = best_valid_error
                best_epoch = epoch

                # TODO: Don't add time needed to save model to training time
                network.save(filename)

                result_str = 'Optimization complete. Best validation error of %.2f %% obtained at epoch %i\n' % (best_valid_error * 100., best_epoch + 1)
            elif (curr_tr_mean < best_train_error):
                best_train_error = curr_tr_mean
                r_val = best_train_error
                best_epoch = epoch

                network.save(filename)
                result_str = 'Optimization complete. Best train error of %.2f %% obtained at epoch %i\n' % (best_train_error * 100., best_epoch + 1)
            else:
                pass

        end_time = timeit.default_timer()

        if (logging):
            sys.stdout.write(result_str)
            sys.stdout.write(('Training took %.2fm\n' % ((end_time - start_time) / 60.)))
            sys.stdout.flush()

        # This should probably be done somewhere else
#        if (test_func):
#            ####################
#            # Final Validation #
#            ####################
#
#            # Resetting to the parameters with the best validation performance
#            network.load(filename)
#
#            sys.stdout.write('... testing the model\n')
#
#            ts_batchinds = np.arange(test_input.shape.eval()[0] // self.batchsize)
#            ts_errors = []
#            for bii, bi in enumerate(ts_batchinds):
#                test_cost, test_error = test_func(bi)
#                ts_errors.append(test_error)
#
#            test_error = np.mean(ts_errors)
#
#            sys.stdout.write(('Test set performance: %.2f %%\n') % (test_error * 100.))
#            sys.stdout.flush()

        return r_val

    def predict(self, network, test_input, filename):
        input = test_input.type()
        index = T.lscalar()
        
        predictions = self.get_predictions(network, input)

        # TODO: Full batch evaluation
        pred_batchsize = self.batchsize
        pred_func = theano.function(inputs=[index],
                                    outputs=[predictions],
                                    givens={input:test_input[index*pred_batchsize:(index+1)*pred_batchsize]},
                                    allow_input_downcast=True)

        #####################
        # Predicting labels #
        #####################

        sys.stdout.write('... predicting for new input\n')

        pred_batchinds = np.arange(test_input.shape.eval()[0] // pred_batchsize)

        test_output = []
        for bii, bi in enumerate(pred_batchinds):
            test_output.append(pred_func(bi))

        np.savez_compressed(filename, test_output=np.array(test_output).flatten())

#    def pretrain(self, network=None, epochs=10, pretrain_input=None, pretrain_output=None, filename=None):
#        """Implements greedy layerwise pre-training as discussed in [1].
#        May be used for stacked autoencoders by setting input=output and
#        defining an appropriate cost function.
#    
#        References:
#            [1] Goodfellow, Ian et al. 
#            "Deep Learning." 
#            MIT Press, 2016
#        """
#
#        if (None in [network, pretrain_input, pretrain_output):
#            raise ValueError('Received incorrect parameters')
#
#        ###########################
#        # Pretraining the network #
#        ###########################
#
#        layer_stack = network.layers
#        network.layers = []
#        pretrained_layers = []
#
#        sys.stdout.write('... pretraining\n')
#
#        for i in xrange(len(layer_stack)):
#            network.layers = layer_stack[0]
#            layer_stack = layer_stack[1:]
#
#            self.train(self, network, train_input=pretrain_input, train_output=pretrain_output, filename=filename)
#            pretrained_layers.append(network.layers)
                                                

    def set_params(self, alpha=0.001, beta1=0.9, beta2=0.999, l1_weight=0.0, l2_weight=0.1):
        alpha=alpha; beta1=beta1; beta2=beta2; l1_weight=l1_weight
        l2_weight=l2_weight
