class SubQuantumCircuit:

    def __init__(self, name, qubits, waiting_qubits):
        self._name = name
        self._qubits = qubits
        self._waiting_qubits = waiting_qubits
        self._total_duration = 0

    @property
    def name(self):
        return self._name

    @property
    def qubits(self):
        return self._qubits

    @property
    def waiting_qubits(self):
        return self._waiting_qubits

    @property
    def total_duration(self):
        return self._total_duration

    def increase_duration(self, amount):
        self._total_duration += amount
