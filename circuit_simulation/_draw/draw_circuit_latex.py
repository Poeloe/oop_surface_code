import re
from circuit_simulation.gates.gate import SingleQubitGate, TwoQubitGate


TWOQUBITGATELOOKUP = {
    'CPhase': ('c-z', 'n-cz'),
    'CNOT': ('cnot', 'n-cnot'),
    'Swap': ('swap', 'swap'),
}


def create_qasm_file(self, meas_error):
    """
        Method constructs a qasm file based on the 'self._draw_order' list. It returns the file path to the
        constructed qasm file.

        Parameters
        ----------
        meas_error : bool
            Specify if there has been introduced an measurement error on purpose to the QuantumCircuit object.
            This is needed to create the proper file name.
    """
    file_path = self._absolute_file_path_from_circuit(meas_error, kind="qasm")
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    file = open(file_path, 'w')

    file.write("\tdef meas,0,'M'\n")
    file.write("\tdef n-meas,0,'\widetilde{M}'\n")
    file.write("\tdef bell,1,'B'\n")
    file.write("\tdef n-bell,1,'\widetilde{B}'\n\n")
    file.write("\tdef n-cnot,1,'\widetilde{X}'\n")
    file.write("\tdef n-cz,1,'\widetilde{Z}'\n")
    file.write("\tdef n-cnot,1,'\widetilde{X}'\n")
    file.write("\tdef n-x,0,'\widetilde{X}'\n")
    file.write("\tdef n-h,0,'\widetilde{H}'\n")
    file.write("\tdef n-y,0,'\widetilde{Y}'\n")

    for i in range(len(self._qubit_array)):
        file.write("\tqubit " + str(i) + "\n")

    file.write("\n")

    for draw_item in self._draw_order:
        operation = draw_item[0]
        qubits = draw_item[1]
        noise = draw_item[2]

        if type(operation) in [SingleQubitGate]:
            operation = operation.representation

        if type(operation) == str:
            operation = ansi_escape.sub("", operation)
            operation = operation.lower()

        if type(qubits) == tuple:
            if type(operation) == TwoQubitGate:
                operation = TWOQUBITGATELOOKUP[operation.name][1 if noise else 0]
            elif '#' in operation:
                operation = 'bell' if not noise else "n-bell"
            tqubit = qubits[0]
            cqubit = qubits[1]
            file.write("\t{} {},{}\n".format(operation, cqubit, tqubit))
        elif "m" in operation:
            file.write("\tmeasure {}\n".format(qubits[0]))
        elif 'level' in operation.lower():
            pass
        else:
            operation = operation if not noise else "n-" + operation
            file.write("\t{} {}\n".format(operation, qubits))

    file.close()

    return file_path