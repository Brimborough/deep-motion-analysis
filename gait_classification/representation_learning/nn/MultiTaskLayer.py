import numpy as np
import theano
import theano.tensor as T

class MultiTaskLayer(object):

    def __init__(self, *networks, **kw):
        self.networks = networks

        if kw.get('params', None) is None:
            self.params = sum([n.params for n in self.networks], [])

    def __call__(self, input):
        """ As each branch may return one or more variables,
            a MultitaskLayer returns a list of results. Each
            Result may, in turn, be a list itself.

            https://en.wikipedia.org/wiki/List_of_lists_of_lists 
        """

        branch_results = []
        for n in self.networks: 
            branch_results.append(n(input))

        return branch_results
        
    def inv(self, output):
        # Where each output comes from a single network
        outputs = []

        for n in self.networks: 
            outputs.append(n.inv(output))

        # Returning the sum by convention
        return reduce(lambda x, y: x+y, outputs)
        
    def load(self, filename):
        for n in self.networks: n.load(filename)

    def save(self, filename): 
        for n in self.networks: n.save(filename)

    def reset(self): 
        for n in self.networks: n.reset()
