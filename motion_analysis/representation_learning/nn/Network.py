import numpy
import theano
import theano.tensor as T

class Network(object):

    def __init__(self, *layers, **kw):
        self.layers = layers
        
        if kw.get('params', None) is None:
            self.params = sum([layer.params for layer in self.layers], [])
        else:
            self.params = kw.get('params', None)
        
    def __call__(self, input):
        for layer in self.layers: input = layer(input)
        return input
    
    def inv(self, output):
        for layer in self.layers[::-1]: output = layer.inv(output)
        return output
    
    def save(self, filename):
        if filename is None: return
        for fname, layer in zip(filename, self.layers):
            layer.save(fname)
        
    def load(self, filename):
        if filename is None: return
        for fname, layer in zip(filename, self.layers):
            layer.load(fname)
        
class InverseNetwork(object):

    def __init__(self, network):
        self.network = network
        self.params = network.params
        
    def __call__(self, input):
        return self.network.inv(input)
    
    def inv(self, output):
        return self.network(output)
    
    def save(self, filename): self.network.save(filename)
    def load(self, filename): self.network.load(filename)
        
        
class AutoEncodingNetwork(object):

    def __init__(self, network):
        self.network = network
        self.params = network.params
        
    def __call__(self, input):
        return self.network.inv(self.network(input))
    
    def inv(self, output):
        return self.network(self.network.inv(output))
    
    def save(self, filename): self.network.save(filename)
    def load(self, filename): self.network.load(filename)
