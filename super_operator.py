import csv
import numpy as np
import pandas as pd
import re
from pprint import pprint


class SuperOperator:

    def __init__(self, error_list_name):
        self._error_list_name = error_list_name

        # Filled by the _convert_error_list method
        self.sup_op_elements_p = []
        self.sup_op_elements_s = []

        self._convert_error_list()

    def __repr__(self):
        return "Superoperator ({})".format(self._error_list_name)

    def __str__(self):
        return "Superoperator ({})".format(self._error_list_name)

    def _convert_error_list(self):

        with open(self._error_list_name) as file:
            reader = pd.read_csv(file, sep=";")

            for i in range(len(list(reader.p_prob))):
                sup_op_el_p = SuperOperatorElement(float(reader.p_prob[i].replace(',', '.')), int(reader.p_lie[i]),
                                                   [ch for ch in reader.p_error[i]])
                sup_op_el_s = SuperOperatorElement(float(reader.s_prob[i].replace(',', '.')), int(reader.s_lie[i]),
                                                   [ch for ch in reader.s_error[i]])
                self.sup_op_elements_p.append(sup_op_el_p)
                self.sup_op_elements_s.append(sup_op_el_s)

        if np.round(np.sum(self.sup_op_elements_p), 6) != 1.0 or np.round(np.sum(self.sup_op_elements_s), 6) != 1.0:
            raise ValueError("Expected joint probabilities of the superoperator to add up to one, instead it was {} for"
                             "the plaquette errors (difference = {}) and {} for the star errors (difference = {}). "
                             "Check your superoperator csv."
                             .format(np.sum(self.sup_op_elements_p), 1.0-np.sum(self.sup_op_elements_p),
                                     np.sum(self.sup_op_elements_s), 1.0-np.sum(self.sup_op_elements_s)))


class SuperOperatorElement:

    def __init__(self, p, lie, error_array):
        self.p = p
        self.lie = lie
        self.error_array = error_array

    def __repr__(self):
        return "SuperOperatorElement(p:{}, lie:{}, errors:{})".format(self.p, self.lie, self.error_array)

    def __str__(self):
        return "SuperOperatorElement(p:{}, lie:{}, errors:{})".format(self.p, self.lie, self.error_array)

    def __ge__(self, other):
        return self.p >= other.p

    def __gt__(self, other):
        return self.p > other.p

    def __le__(self, other):
        return self.p <= other.p

    def __lt__(self, other):
        return self.p < other.p

    def __add__(self, other):
        return self.p + other.p

    def __radd__(self, other):
        return self.p + other
