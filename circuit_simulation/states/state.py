import scipy.sparse as sp
import numpy as np
import re


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
        self._sp_vector.eliminate_zeros()
        return self._sp_vector

    def __repr__(self):
        return self.representation

    def __eq__(self, other):
        if type(other) != State:
            return False
        return np.array_equal(self.vector, other.vector)

    def __rmul__(self, other):
        if type(other) not in [int, float]:
            return ValueError("Not allowed! Please multiply by int or float")
        return State(self.name, other * self.vector, '{}({})'.format(other, self.representation))

    def __mul__(self, other):
        if type(other) != State:
            raise ValueError("Not allowed! If multiplying by a factor is wanted, please put it in front.")
        pattern = re.compile('(?<=\|)(.*)(?=\>)')
        first_repr = re.search(pattern, self.representation).group(1)
        second_repr = re.search(pattern, other.representation).group(1)
        vector = sp.kron(self.sp_vector, other.sp_vector)
        vector.eliminate_zeros()
        return State(self.name + other.name,
                     vector,
                     "|{}{}>".format(first_repr, second_repr))

    def __add__(self, other):
        vector = self.sp_vector + other.sp_vector
        repr_state = '{} + {}'.format(self.representation, other.representation)
        return State(repr_state, vector, repr_state)

