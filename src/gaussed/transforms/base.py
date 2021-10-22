"""Transform objects act as a convenience wrapper for the transformations on, for instance, the Mean, Kernel or Basis
objects that actually get transformed. The actual transformations act on the transformer attribute of a Distribution 
class which is defined relative to the Distribution object. For instance, in the HilbertGP class, the underyling
transformer attribute would be an instance of the Laplace basis object. The logic of the transformations are defined 
in the corresponding class of the transformer instance. """


class Transform:
    def __init__(self, distribution):
        super().__init__()
        self._distribution = distribution

        # multi_input is True if input of Transform is an arbitrary number of Distributions, an integer value
        # if Transform takes a given number of Distributions or False if only takes one Distribution.
        self.multi_input = None
        # Similarly with multi_output, but now Transform outputs a number of Distributions
        self.multi_output = None

    def forward(self, inputs):
        raise NotImplementedError()
