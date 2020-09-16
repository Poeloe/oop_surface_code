import re
from termcolor import colored, COLORS
from circuit_simulation.gates.gate import SingleQubitGate, TwoQubitGate
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

        if (diff := len(b_stripped) - len(a_stripped)) > 0:
            state_repr_split = init_state_repr[a[0]].split(" ")
            init_state_repr[a[0]] = state_repr_split[0] + ((diff + 1) * " ") + state_repr_split[1]
        elif (diff := len(a_stripped) - len(b_stripped)) > 0:
            state_repr_split = init_state_repr[b[0]].split(" ")
            init_state_repr[b[0]] = state_repr_split[0] + ((diff + 1) * " ") + state_repr_split[1]

    return init_state_repr


def draw_gates(self, init, no_color):
    """ Adds the visual representation of the operations applied on the qubits """
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    for draw_item in self._draw_order:
        gate = draw_item[0]
        qubits = draw_item[1]
        noise = draw_item[2]

        if type(qubits) == tuple:
            if type(gate) in [SingleQubitGate, TwoQubitGate]:
                control = gate.control_repr if type(gate) == TwoQubitGate else "o"
                gate = gate.representation
            elif gate == "#":
                control = gate
            else:
                control = "o"

            if noise:
                control = "~" + control if no_color else colored("~", 'red') + control
                gate = "~" + gate if no_color else colored('~', 'red') + gate

            cqubit = qubits[0]
            tqubit = qubits[1]
            init[cqubit] += "---{}---".format(control)
            init[tqubit] += "---{}---".format(gate)
        else:
            if type(gate) == SingleQubitGate:
                gate = gate.representation
            if noise:
                gate = "~" + gate if no_color else colored("~", 'red') + gate
            init[qubits] += "---{}---".format(gate)

        for a, b in it.combinations(enumerate(init), 2):
            # Since colored ansi code is shown as color and not text it should be stripped for length comparison
            a_stripped = ansi_escape.sub("", init[a[0]])
            b_stripped = ansi_escape.sub("", init[b[0]])

            if (diff := len(b_stripped) - len(a_stripped)) > 0:
                init[a[0]] += diff * "-"
            elif (diff := len(a_stripped) - len(b_stripped)) > 0:
                init[b[0]] += diff * "-"


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


def add_draw_operation(self, operation, qubits, noise=False):
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

    if type(qubits) is tuple:

        cqubit = qubits[0] + self._effective_measurements
        tqubit = qubits[1] + self._effective_measurements

        if self._measured_qubits != [] and cqubit >= min(self._measured_qubits):
            cqubit += len(self._measured_qubits)
        if self._measured_qubits != [] and tqubit >= min(self._measured_qubits):
            tqubit += len(self._measured_qubits)

        qubits = (cqubit, tqubit)
    else:
        qubits += int(self._effective_measurements)

        if self._measured_qubits != [] and qubits >= min(self._measured_qubits):
            qubits += len(self._measured_qubits)
    item = [operation, qubits, noise]
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