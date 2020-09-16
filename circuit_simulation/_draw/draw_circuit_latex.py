import re
from circuit_simulation.gates.gate import SingleQubitGate, TwoQubitGate


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
        gate = draw_item[0]
        qubits = draw_item[1]
        noise = draw_item[2]

        if type(gate) in [SingleQubitGate, TwoQubitGate]:
            gate = gate.representation

        gate = ansi_escape.sub("", gate)
        gate = gate.lower()
        if type(qubits) == tuple:
            if 'z' in gate:
                gate = "c-z" if not noise else "n-cz"
            elif 'x' in gate:
                gate = 'cnot' if not noise else "n-cnot"
            elif '#' in gate:
                gate = 'bell' if not noise else "n-bell"
            cqubit = qubits[0]
            tqubit = qubits[1]
            file.write("\t" + gate + " " + str(cqubit) + "," + str(tqubit) + "\n")
        elif "m" in gate:
            gate = "meas " if "~" not in gate else "n-meas "
            file.write("\t" + gate + str(qubits) + "\n")
        else:
            gate = gate if "~" not in gate or not noise else "n-" + gate
            file.write("\t" + gate + " " + str(qubits) + "\n")

    file.close()

    return file_path