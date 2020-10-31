import scipy.sparse as sp
import numpy as np

class State(object):
    """
        State class

        Class defines a state with the attributes

        Attributes
        ----------
        name : str
            Name that will be used to refer to the state
        vector : numpy array
            Vector that represents the state
        representation : str
            Representation is used in the drawing of the circuits
    """

    def __init__(self, name, vector, representation):
        self._name = name
        self._vector = vector
        self._representation = representation
        self._sp_vector = sp.csr_matrix(vector)

    @property
    def name(self):
        return self._name

    @property
    def vector(self):
        return self._vector

    @property
    def representation(self):
        return self._representation

    @property
    def sp_vector(self):
        return self._sp_vector

    def __repr__(self):
        return self.representation

    def __eq__(self, other):
        if type(other) != State:
            return False
        return np.array_equal(self.vector, other.vector)