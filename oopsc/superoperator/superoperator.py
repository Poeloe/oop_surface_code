import numpy as np
import pandas as pd
import os
import random
import copy


class Superoperator:

    def __init__(self, file_name, GHZ_success=1.1):
        """
            Superoperator(file_name, graph, GHZ_success=1.1)

                Superoperator object that contains a list of SuperoperatorElements for both stabilizer types
                (plaquette and star) that specifies what errors occur on the stabilizer qubits and the
                corresponding probability of the that error occurring.

                Parameters:
                -----------
                file_name : str
                    File name of the csv file that specifies the errors on the stabilizer qubits. File must be
                    placed in the 'csv_files' folder.
                graph : graph object
                    The graph object that creates the Superoperator object. This is necessary to sort the existing
                    stabilizer per round
                GHZ_success : float [0-1], optional, default=1.1
                    The percentage of stabilizers that are successfully created by the protocol that the superoperator
                    is the result of.

                Attributes:
                -----------
                file_name : str
                    File name of the csv file that specifies the errors on the stabilizer qubits. File must be
                    placed in the 'csv_files' folder.
                GHZ_success : float [0-1], optional, default=1.1
                    The percentage of stabilizers that are successfully created by the protocol that the superoperator
                    is the result of.
                sup_op_elements_p : list
                    The list of SuperoperatorElement objects that specifies the errors and their probabilities
                    occurring on the plaquette stabilizers
                sup_op_elements_s : list
                    The list of SuperoperatorElement objects that specifies the errors and their probabilities
                    occurring on the star stabilizers
                stabs_p1 : dict
                    A dictionary with the z layer a key and the value a list of the stabilizers that are involved in
                    the first round of plaquette stabilizer measurements for that layer.
                stabs_p2 : dict
                    A dictionary with the z layer a key and the value a list of the stabilizers that are involved in
                    the second round of plaquette stabilizer measurements for that layer.
                stabs_s1 : dict
                    A dictionary with the z layer a key and the value a list of the stabilizers that are involved in
                    the first round of star stabilizer measurements for that layer.
                stabs_s2 : dict
                    A dictionary with the z layer a key and the value a list of the stabilizers that are involved in
                    the second round of star stabilizer measurements for that layer.
        """
        self.file_name = file_name
        self.GHZ_success = GHZ_success

        self.pg = 0
        self.pm = 0
        self.pn = 0

        # Filled by the _convert_error_list method
        self.sup_op_elements_p = []
        self.sup_op_elements_s = []
        self.sup_op_elements_p2 = []
        self.sup_op_elements_s2 = []

        # For speed up purposes, the superoperator has the stabilizers split into rounds as attributes
        self.stabs_p1, self.stabs_p2, self.stabs_s1, self.stabs_s2 = {}, {}, {}, {}

        # self._get_stabilizer_rounds(graph)
        self._convert_error_list()
        self._convert_second_round_elements()


    def __repr__(self):
        return "Superoperator ({})".format(self.file_name)

    def __str__(self):
        return self.__repr__()

    def _convert_error_list(self):
        """
            Retrieves the list of SuperoperatorElements for both the stabilizer types from the supplied csv file
            name.

            CSV file column format
            -----------------------
            p_prob : float
                Column containing the probabilities for the specific errors on the stabilizer qubits.
            p_lie : bool
                Contains the information of whether or not a measurement error occurred.
            p_error : str
                Specify the occurred error configuration
            GHZ_success : float, optional
                Percentage of stabilizers that were able to be created successfully. If not specified, the success
                rate will be set to 1.1
            pm : float, optional
                Measurement error rate
            pg : float, optional
                Gate error rate
            pn : float, optional
                Network error rate

            CSV file format example
            -----------------------
            p_prob;     p_lie;  p_error;    s_prob;    s_lie;   s_error;    GHZ_success;    pg;     pm;    pn;
            0.9509;     0    ;  IIII   ;    0.950 ;    0    ;   IIII   ;    0.99       ;  0.01;   0.01;  0.01;
            0.0384;     0    ;  IIIX   ;    0.038 ;    0    ;   IIIX   ;               ;      ;       ;      ;
        """
        path_to_file = os.path.join(os.path.dirname(__file__), "csv_files", self.file_name + ".csv")

        with open(path_to_file) as file:
            reader = pd.read_csv(file, sep=";")

            # If GHZ_success is 1.1 it has obtained the default value and can be overwritten
            if 'GHZ_success' in reader and self.GHZ_success == 1.1:
                self.GHZ_success = float(str(reader.GHZ_success[0]).replace(',', '.').replace(" ", ""))
            if "pm" in reader:
                self.pg = float(str(reader.pm[0]).replace(",", ".").replace(" ", ""))
            if "pg" in reader:
                self.pm = float(str(reader.pg[0]).replace(",", ".").replace(" ", ""))
            if "pn" in reader:
                self.pn = float(str(reader.pn[0]).replace(",", ".").replace(" ", ""))

            for i in range(len(list(reader.p_prob))):
                # Do some parsing operations on the entries to ensure proper form
                p_prob = float(str(reader.p_prob[i]).replace(',', '.').replace(" ", ""))
                s_prob = float(str(reader.s_prob[i]).replace(',', '.').replace(" ", ""))
                p_error = [ch for ch in reader.p_error[i].replace(" ", "")]
                s_error = [ch for ch in reader.s_error[i].replace(" ", "")]

                self.sup_op_elements_p.append(SuperoperatorElement(p_prob, bool(int(reader.p_lie[i])), p_error))
                self.sup_op_elements_s.append(SuperoperatorElement(s_prob, bool(int(reader.s_lie[i])), s_error))

        # Check if the probabilities add up to 1 to ensure a valid decomposition
        if round(sum(self.sup_op_elements_p), 6) != 1.0 or round(sum(self.sup_op_elements_s), 6) != 1.0:
            raise ValueError("Expected joint probabilities of the superoperator to add up to one, instead it was {} for"
                             "the plaquette errors (difference = {}) and {} for the star errors (difference = {}). "
                             "Check your superoperator csv."
                             .format(sum(self.sup_op_elements_p), 1.0 - sum(self.sup_op_elements_p),
                                     sum(self.sup_op_elements_s), 1.0 - sum(self.sup_op_elements_s)))

        # Sort the entries such that the most likely entries will be listed first
        self.sup_op_elements_p = sorted(self.sup_op_elements_p, reverse=True)
        self.sup_op_elements_s = sorted(self.sup_op_elements_s, reverse=True)

    def _convert_second_round_elements(self):
        self.sup_op_elements_p2 = copy.deepcopy(self.sup_op_elements_p)
        self.sup_op_elements_s2 = copy.deepcopy(self.sup_op_elements_s)

        for sup_op_el_p2, sup_op_el_s2 in zip(self.sup_op_elements_p2, self.sup_op_elements_s2):
            if (sup_op_el_p2.error_array.count("I") + sup_op_el_p2.error_array.count("Z")) % 2 == 1:
                sup_op_el_p2.lie = not sup_op_el_p2.lie
            if (sup_op_el_s2.error_array.count("I") + sup_op_el_s2.error_array.count("X")) % 2 == 1:
                sup_op_el_s2.lie = not sup_op_el_s2.lie

    def _get_stabilizer_rounds(self, graph, z=0):
        """
            Obtain for both type of stabilizers the stabilizers that will be measured each round for every
            measurement layer z. These rounds are necessary when non local stabilizer measurements protocols
            are used.

            Parameters
            ----------
            graph : graph object
                The graph object that the Superoperator object is applied to
            z : int, optional, default=0
                The measurement layer for which the stabilizers should be divided in rounds
        """
        if z in self.stabs_p1:
            return
        else:
            self.stabs_p1[z] = []
            self.stabs_p2[z] = []
            self.stabs_s1[z] = []
            self.stabs_s2[z] = []

        for stab in graph.S[z].values():
            even_odd = stab.sID[1] % 2
            if stab.sID[2] % 2 == even_odd:
                if stab.sID[0] == 0:
                    self.stabs_s1[z].append(stab)
                else:
                    self.stabs_p1[z].append(stab)
            else:
                if stab.sID[0] == 0:
                    self.stabs_s2[z].append(stab)
                else:
                    self.stabs_p2[z].append(stab)

    def set_stabilizer_rounds(self, graph, z):
        self._get_stabilizer_rounds(graph, z=z)

    @staticmethod
    def get_supop_el_by_prob(superoperator_elements):
        """
            Retrieve a SuperoperatorElement from a list of SuperoperatorElements based on the probabilities of
            these SuperoperatorElements. This means, that the method is more likely to return a SuperoperatorElement
            with a high probability than one with a low probability.

            Parameters
            ----------
            superoperator_elements : list
                List containing SuperoperatorElements of which a SuperoperatorElement should be picked

            Returns
            -------
            superoperator_element : SuperoperatorElement
                A SuperoperatorElement is returned from the superoperator_elements list based on the probability
        """
        r = random.random()
        index = 0
        while r >= 0 and index < len(superoperator_elements):
            r -= superoperator_elements[index].p
            index += 1
        return superoperator_elements[index - 1]


class SuperoperatorElement:

    def __init__(self, p, lie, error_array):
        """
            SuperoperatorElement(p, lie, error_array)

                Used as building block for the Superoperator object. It contains the error configuration on the
                stabilizer qubits, the presents of a measurement error and the corresponding probability.

                Parameters
                ----------
                p : float
                    Probability of the specific error configurations occurring on the stabilizer qubits
                lie : bool
                    Whether the a measurement error is involved.
                error_array : list
                    List of four characters that represent Pauli errors occurring on a qubit. One can choose from
                    'X', 'Y', 'Z' or 'I'.
        """
        self.p = p
        self.lie = lie
        self.error_array = error_array
        self.id = str(p) + str(lie) + str(error_array)

    def __repr__(self):
        return "SuperOperatorElement(p:{}, lie:{}, errors:{})".format(self.p, self.lie, self.error_array)

    def __str__(self):
        return self.__repr__()

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

    @staticmethod
    def file_path():
        return str(os.path.dirname(__file__))

    def full_equals(self, other, rnd=8, sort_array=True):
        if sort_array:
            self.error_array.sort()
            other.error_array.sort()

        return (round(self.p, rnd) == round(other.p, rnd)
                and self.lie == other.lie
                and self.error_array == other.error_array)

    def error_array_lie_equals(self, other, sort_array=True):
        if sort_array:
            self.error_array.sort()
            other.error_array.sort()

        return self.lie == other.lie and self.error_array == other.error_array

    def probability_lie_equals(self, other, rnd=8):
        return round(self.p, rnd) == round(other.p, rnd) and self.lie == other.lie
