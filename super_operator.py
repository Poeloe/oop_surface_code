import csv
import numpy as np
from pprint import pprint


class SuperOperator:

    def __init__(self, error_list_name):
        self._error_list_name = error_list_name

        # Filled by the _convert_error_list method
        self.sup_op_elements = []

        self._convert_error_list()

    def __repr__(self):
        return "Superoperator ({})".format(self._error_list_name)

    def __str__(self):
        return "Superoperator ({})".format(self._error_list_name) + str(self.errors)

    def _convert_error_list(self):

        with open(self._error_list_name) as file:
            reader = csv.reader(file)

            for i, row in enumerate(reader):
                if i == 0:
                    if len(row) != 3:
                        raise TypeError("CSV not of expected format")
                    continue

                sup_op = SuperOperatorElement(float(row[0]), int(row[1]), [ch for ch in row[2]])
                self.sup_op_elements.append(sup_op)

    def get_probabilities(self):
        probs = []
        for sup_op_el in self.sup_op_elements:
            probs.append(sup_op_el.p)

        return probs


class SuperOperatorElement:

    def __init__(self, p, lie, error_array):
        self.p = p
        self.lie = lie
        self.error_array = error_array

    def __repr__(self):
        return "SuperOperatorElement(p:{}, lie:{}, errors:{})".format(self.p, self.lie, self.error_array)

    def __str__(self):
        return "SuperOperatorElement(p:{}, lie:{}, errors:{})".format(self.p, self.lie, self.error_array)


if __name__ == "__main__":
    sup_op = SuperOperator("error_list_test.csv")
    pprint(sup_op.errors)