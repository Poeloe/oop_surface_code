

class Node(object):

    def __init__(self, name: str, qubits: list, qc, electron_qubits=None, data_qubits=None,
                 ghz_qubit=None):

        self._name = name
        self._qubits = qubits
        self._qc = qc
        self._electron_qubits = electron_qubits
        self._data_qubits = data_qubits
        self._ghz_qubit = ghz_qubit

    @property
    def name(self):
        return self._name

    @property
    def qubits(self):
        return self._qubits

    @property
    def qc(self):
        return self._qubits

    @property
    def electron_qubits(self):
        return self._electron_qubits

    @property
    def data_qubits(self):
        return self._data_qubits

    @property
    def ghz_qubit(self):
        return self._ghz_qubit

    def qubit_in_node(self, qubit):
        return qubit in self.qubits
