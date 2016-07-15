import numpy as np
import theano
import theano.tensor as T

from Network import Network

class Ensemble(object):

    def __init__(self, batchsize, networks=[]):

        if ([] == networks):
            raise ValueError('Received empty list of networks')

        self.networks = networks
        self.batchsize = batchsize

        self.y_pred = lambda network, x: network(x)
        self.pred   = lambda Y: np.argmax(Y, axis=1)
        self.error  = lambda y_pred, y_output: np.mean(y_pred != y_output)

    def get_predictions(self, network, input):
        return self.y_pred(network, input)

    def evaluate_network(self, network, eval_input, eval_output):
        input = eval_input.type()
        index = T.lscalar()

        predictions = self.get_predictions(network, input)

        pred_func = theano.function(inputs=[index],
                                    outputs=[predictions],
                                    givens={input:eval_input[index*self.batchsize:(index+1)*self.batchsize]},
                                    allow_input_downcast=True)

        pred_batchinds = np.arange(eval_input.shape.eval()[0] // self.batchsize)

        network_output = []
        for bii, bi in enumerate(pred_batchinds):
            network_output.append(pred_func(bi))

        return np.array(network_output).squeeze().reshape(10000, 10)

    def eval(self, eval_input, eval_output):

        true_labels = self.pred(eval_output.eval())

        print '---------------------------------'

        network_results = []
        best_performance = np.inf

        for id, n in enumerate(self.networks):
            n_result = self.evaluate_network(n, eval_input, eval_output)
            performance = self.error(self.pred(n_result), true_labels)

            if (performance < best_performance):
                best_performance = performance

            print 'Performance of Network %i: %.5f' % (id+1, performance)
            network_results.append(n_result)

        # Average network predicitons
        ens_predictions = self.pred(np.mean(network_results, axis=0))
        ens_performance = self.error(ens_predictions, true_labels)
        diff = (ens_performance - best_performance)
        print '---------------------------------'
        print 'Performance of Ensemble: %.5f' % (ens_performance)
        print 'Diff to best Network: %.5f' % (diff)
