import numpy as np
import pandas as pd


class SuperOperator:

    def __init__(self, file_name, graph):
        self.file_name = file_name

        # Filled by the _convert_error_list method
        self.sup_op_elements_p = []
        self.sup_op_elements_s = []
        self.weights_p = []
        self.weights_s = []

        # For speed up purposes, the superoperator has the stabilizers split into rounds as attributes
        self.stabs_p1, self.stabs_p2, self.stabs_s1, self.stabs_s2 = [], [], [], []

        self._get_stabilizer_rounds(graph)
        self._convert_error_list()

    def __repr__(self):
        return "Superoperator ({})".format(self.file_name)

    def __str__(self):
        return "Superoperator ({})".format(self.file_name)

    def _convert_error_list(self):

        with open(self.file_name) as file:
            reader = pd.read_csv(file, sep=";")

            for i in range(len(list(reader.p_prob))):
                p_prob = float(reader.p_prob[i].replace(',', '.'))
                s_prob = float(reader.s_prob[i].replace(',', '.'))
                sup_op_el_p = SuperOperatorElement(p_prob, int(reader.p_lie[i]), [ch for ch in reader.p_error[i]])
                sup_op_el_s = SuperOperatorElement(s_prob, int(reader.s_lie[i]), [ch for ch in reader.s_error[i]])
                self.sup_op_elements_p.append(sup_op_el_p)
                self.sup_op_elements_s.append(sup_op_el_s)

                self.weights_p.append(p_prob)
                self.weights_s.append(s_prob)

        if round(sum(self.sup_op_elements_p), 6) != 1.0 or round(sum(self.sup_op_elements_s), 6) != 1.0:
            raise ValueError("Expected joint probabilities of the superoperator to add up to one, instead it was {} for"
                             "the plaquette errors (difference = {}) and {} for the star errors (difference = {}). "
                             "Check your superoperator csv."
                             .format(sum(self.sup_op_elements_p), 1.0-sum(self.sup_op_elements_p),
                                     sum(self.sup_op_elements_s), 1.0-sum(self.sup_op_elements_s)))

    def _get_stabilizer_rounds(self, graph, stab_type=None, z=0):
        # if stab_type is not None:
        #     if re.match('.*star.*|[0]', str(stab_type), flags=re.IGNORECASE) is not None:
        #         stab_type = 0
        #     elif re.match('.*plaq.*|[1]', str(stab_type), flags=re.IGNORECASE) is not None:
        #         stab_type = 1
        #     else:
        #         raise ValueError("No valid stabilizer type provided, expected 'star' or 'plaquette'.")

        stabs_p = []
        stabs_s = []

        for stab in graph.S[z].values():
            even_odd = stab.sID[1] % 2
            if stab.sID[2] % 2 == even_odd:
                if stab.sID[0] == 0:
                    self.stabs_s1.append(stab)
                else:
                    self.stabs_p1.append(stab)
            else:
                if stab.sID[0] == 0:
                    self.stabs_s2.append(stab)
                else:
                    self.stabs_p2.append(stab)

    def set_stabilizer_rounds(self, z):
        self._get_stabilizer_rounds(z=z)


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
