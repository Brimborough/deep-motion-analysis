class Param(object):
    """ Represents a wrapper for shared variables such as weights and
        biases. This allows to store useful information with them.
    """
    def __init__(self, value, regularisable=True):
        self.p_value = value
        self.regularisable = regularisable
