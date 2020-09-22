from copy import copy


class SubQuantumCircuit:

    def __init__(self, name, qubits, waiting_qubits, concurrent_sub_circuits=None):
        self._name = name
        self._qubits = qubits
        self._waiting_qubits = waiting_qubits
        self._total_duration = 0
        self._concurrent_sub_circuits = concurrent_sub_circuits if concurrent_sub_circuits is not None else []

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

    @property
    def concurrent_sub_circuits(self):
        return self._concurrent_sub_circuits

    @property
    def get_all_concurrent_qubits(self):
        total_qubits = copy(self.qubits)
        for sub_circuit in self.concurrent_sub_circuits:
            total_qubits.extend(sub_circuit.qubits)

        return total_qubits

    def increase_duration(self, amount):
        self._total_duration += amount

    def add_concurrent_sub_circuits(self, sub_circuits):
        if type(sub_circuits) != list:
            sub_circuits = [sub_circuits]
        for sub_circuit in sub_circuits:
            self._concurrent_sub_circuits.append(sub_circuit)

        self._concurrent_sub_circuits = list(set(self._concurrent_sub_circuits))

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        if type(other) != SubQuantumCircuit:
            raise ValueError("It is not possible to compare a SubQuantumCircuit object with anything else!")

        return self.name < other.name
