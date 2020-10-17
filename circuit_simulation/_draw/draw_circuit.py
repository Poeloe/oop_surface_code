import re
from circuit_simulation.termcolor.termcolor import colored, COLORS
from circuit_simulation.gates.gate import SingleQubitGate, TwoQubitGate
from circuit_simulation.sub_circuit.sub_quantum_circuit import SubQuantumCircuit
import numpy as np
from circuit_simulation.utilities.decorators import handle_none_parameters
import itertools as it


def draw_init(self, no_color):
    """ Returns an array containing the visual representation of the initial state of the qubits. """
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    init_state_repr = []
    for qubit, state in enumerate(self._qubit_array):
        node_name = self.get_node_name_from_qubit(qubit)
        init_state_repr.append("\n\n{}{} ---".format(node_name + ":" if node_name is not None else "",
                                                     ansi_escape.sub("", state.representation) if no_color else
                                                     state.representation))

    for a, b in it.combinations(enumerate(init_state_repr), 2):
        # Since colored ansi code is shown as color and not text it should be stripped for length comparison
        a_stripped = ansi_escape.sub("", init_state_repr[a[0]])
        b_stripped = ansi_escape.sub("", init_state_repr[b[0]])

        if len(b_stripped) - len(a_stripped) > 0:
            diff = len(b_stripped) - len(a_stripped)
            state_repr_split = init_state_repr[a[0]].split(" ")
            init_state_repr[a[0]] = state_repr_split[0] + ((diff + 1) * " ") + state_repr_split[1]
        elif len(a_stripped) - len(b_stripped) > 0:
            diff = len(a_stripped) - len(b_stripped)
            state_repr_split = init_state_repr[b[0]].split(" ")
            init_state_repr[b[0]] = state_repr_split[0] + ((diff + 1) * " ") + state_repr_split[1]

    return init_state_repr


def draw_gates(self, init, no_color):
    """ Adds the visual representation of the operations applied on the qubits """
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    for draw_item in self._draw_order:
        # A level item sets the length of al qubit paths the same. This is usually used for points where a sub
        # circuit waits on another sub circuit to finish before continuing with the rest of the circuit
        if draw_item == "LEVEL":
            init = _level_qubit_paths(init)
            continue
        gate = draw_item[0]
        qubits = draw_item[1]
        noise = draw_item[2]
        sub_circuit = draw_item[3]
        sub_circuit_concurrent = draw_item[4]
        if sub_circuit_concurrent:
            concurrent_qubits = sub_circuit.qubits if sub_circuit is not None else []
        else:
            concurrent_qubits = sub_circuit.get_all_concurrent_qubits if sub_circuit is not None else []

        # Find qubits that are not involved in the current sub circuit
        non_involved_qubits = list(set(concurrent_qubits) ^ set([i for i in range(self.num_qubits)]))

        # Draw 2 qubit operations
        if len(qubits) == 2:
            if type(gate) in [SingleQubitGate, TwoQubitGate]:
                control = gate.control_repr if type(gate) == TwoQubitGate else "o"
                gate = gate.representation
            elif "#" in gate:
                control = gate
            else:
                control = "o"

            if noise:
                control = "~" + control if no_color else colored("~", 'red') + control
                gate = "~" + gate if no_color else colored('~', 'red') + gate

            cqubit = qubits[1]
            tqubit = qubits[0]

            init = _correct_path_length(init, cqubit, tqubit)

            init[cqubit] += "---{}---".format(control)
            init[tqubit] += "---{}---".format(gate)

            cqubit_stripped = ansi_escape.sub("", init[cqubit])
            tqubit_stripped = ansi_escape.sub("", init[tqubit])
            longest_item = cqubit_stripped if len(cqubit_stripped) >= len(tqubit_stripped) else tqubit_stripped
            current_qubits = list(set(self.get_node_qubits(cqubit) + self.get_node_qubits(tqubit)
                                      + non_involved_qubits))
        else:
            if type(gate) == SingleQubitGate:
                gate = gate.representation
            if noise:
                gate = "~" + gate if no_color else colored("~", 'red') + gate
            init[qubits[0]] += "---{}---".format(gate)

            longest_item = ansi_escape.sub("", init[qubits[0]])
            current_qubits = list(set(self.get_node_qubits(qubits[0]) + non_involved_qubits))

        partial_update = list(np.array(init)[current_qubits]) if current_qubits != [] else init
        for index, item in enumerate(partial_update):
            item_stripped = ansi_escape.sub("", item)
            diff = len(longest_item) - len(item_stripped)
            if current_qubits != []:
                init[current_qubits[index]] += diff * "-"
            else:
                init[index] += diff * '-'


def _correct_path_length(init, qubit_1, qubit_2):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    len_qubit_1 = len(ansi_escape.sub("", init[qubit_1]))
    len_qubit_2 = len(ansi_escape.sub("", init[qubit_2]))

    if len_qubit_1 > len_qubit_2:
        diff = len_qubit_1 - len_qubit_2
        init[qubit_2] += diff * "-"
    elif len_qubit_2 > len_qubit_1:
        diff = len_qubit_2 - len_qubit_1
        init[qubit_1] += diff * "-"

    return init


def _level_qubit_paths(init):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    init_lengths = [len(ansi_escape.sub("", item)) for item in init]
    longest_path = max(init_lengths)

    for index, path in enumerate(init):
        path_length = len(ansi_escape.sub("", path))
        diff = longest_path - path_length
        init[index] += diff * "-"

    return init


def color_qubit_lines(self, init):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    colors = sorted(list(COLORS.keys()))
    colors.remove('white'), colors.remove('grey')
    node_color_dict = {}
    for i, key in enumerate(self.nodes.keys()):
        if i == len(colors) - 1:
            self._print_lines.append("Warning! To many nodes for the amount of different colors. Colors are reused")
        color_number = i % (len(colors) - 1)
        node_color_dict[key] = colors[color_number]

    for i, _ in enumerate(init):
        node_name = self.get_node_name_from_qubit(i)
        if node_name is None: continue
        espaced_lines = ansi_escape.sub("", init[i])
        init[i] = colored(espaced_lines, node_color_dict[node_name])


@handle_none_parameters
def add_draw_operation(self, operation, qubits, noise=False, _current_sub_circuit=None, sub_circuit_concurrent=False):
    """
        Adds an operation to the draw order list.

        Notes
        -----
        **Note** :
            Since measurements and additions of qubits change the qubit indices dynamically, this will be
            accounted for in this method when adding a draw operation. The '_effective_measurement' attribute keeps
            track of how many qubits have effectively been measured, which means they have not been reinitialised
            after measurement (by creating a Bell-pair at the top or adding a top qubit). The '_measured_qubits'
            attribute contains all the qubits that have been measured and are not used anymore after (in means of
            the drawing scheme).

        **2nd Note** :
            Please consider that the drawing of the circuit can differ from reality due to this dynamic
            way of changing the qubit indices with measurement and/or qubit addition operations. THIS EFFECTIVELY
            MEANS THAT THE CIRCUIT REPRESENTATION MAY NOT ALWAYS PROPERLY REPRESENT THE APPLIED CIRCUIT WHEN USING
            MEASUREMENTS AND QUBIT ADDITIONS.
    """
    if type(qubits) == int:
        qubits = [qubits]

    if len(qubits) > 1:

        cqubit = qubits[0] + self._effective_measurements
        tqubit = qubits[1] + self._effective_measurements

        if self._measured_qubits != [] and cqubit >= min(self._measured_qubits):
            cqubit += len(self._measured_qubits)
        if self._measured_qubits != [] and tqubit >= min(self._measured_qubits):
            tqubit += len(self._measured_qubits)

        qubits = (cqubit, tqubit)
    else:
        qubits[0] += int(self._effective_measurements)

        if self._measured_qubits != [] and qubits >= min(self._measured_qubits):
            qubits += len(self._measured_qubits)
    item = [operation, qubits, noise, _current_sub_circuit, sub_circuit_concurrent]
    self._draw_order.append(item)


def correct_drawing_for_n_top_qubit_additions(self, n=1):
    """
        Corrects the self._draw_order list for addition of n top qubits.

        When a qubit gets added to the top of the stack, it gets the index 0. This means that the indices of the
        already existing qubits increase by 1. This should be corrected for in the self._draw_order list, since
        the qubit references used the 'old' qubit index.

        *** Note that for the actual qubit operations that already have been applied to the system the addition of
        a top qubit is not of importance, but after addition the user should know this index change for future
        operations ***

        Parameters
        ----------
        n : int, optional, default=1
            Amount of added top qubits that should be corrected for.
    """
    self._measured_qubits.extend([i for i in range(self._effective_measurements)])
    self._measured_qubits = [(x + n) for x in self._measured_qubits]
    self._effective_measurements = 0
    for i, draw_item in enumerate(self._draw_order):
        operation = draw_item[0]
        qubits = draw_item[1]
        noise = draw_item[2]
        if type(qubits) == tuple:
            self._draw_order[i] = [operation, (qubits[0] + n, qubits[1] + n), noise]
        else:
            self._draw_order[i] = [operation, qubits + n, noise]


def correct_drawing_for_circuit_fusion(self, other_draw_order, num_qubits_other):
    new_draw_order = other_draw_order
    for draw_item in self._draw_order:
        operation = draw_item[0]
        if type(draw_item[1]) == tuple:
            qubits = tuple([i + num_qubits_other for i in draw_item[1]])
        else:
            qubits = draw_item[1] + num_qubits_other
        noise = draw_item[2]
        new_draw_item = [operation, qubits, noise]
        new_draw_order.append(new_draw_item)
    self._draw_order = new_draw_order