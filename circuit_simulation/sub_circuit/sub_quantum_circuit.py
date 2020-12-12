from copy import copy


class SubQuantumCircuit:

    def __init__(self, name, qubits, waiting_qubits, concurrent_sub_circuits=None, involved_nodes=None):
        self._name = name
        self._qubits = qubits
        self._waiting_qubits = waiting_qubits
        self._total_duration = 0
        self._concurrent_sub_circuits = concurrent_sub_circuits if concurrent_sub_circuits is not None else []
        self._involved_nodes = involved_nodes if involved_nodes is not None else []
        self._cut_off_time_reached = False
        self._ran = False
        self._concurrent_swap_wait_applied = 0

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
    def involved_nodes(self):
        return self._involved_nodes

    @property
    def amount_involved_nodes(self):
        return len(self._involved_nodes)

    @property
    def amount_concurrent_sub_circuits(self):
        return len(self.concurrent_sub_circuits) + 1

    @property
    def get_all_concurrent_qubits(self):
        total_qubits = copy(self.qubits)
        for sub_circuit in self.concurrent_sub_circuits:
            total_qubits.extend(sub_circuit.qubits)

        return total_qubits

    @property
    def ran(self):
        return self._ran

    @property
    def all_ran(self):
        return all([sc.ran for sc in self._concurrent_sub_circuits])

    @property
    def cut_off_time_reached(self):
        return self._cut_off_time_reached

    def set_cut_off_time_reached(self, value=True):
        if type(value) != bool:
            raise ValueError("Property 'cut_off_time_reached' cannot be set with types other than bool.")
        self._cut_off_time_reached = value

    def set_ran(self, value=True):
        if type(value) != bool:
            raise ValueError("Property 'ran' cannot be set with types other than bool.")
        self._ran = value

    def increase_duration(self, amount):
        self._total_duration += amount

    def increase_amount_concurrent_swap_wait(self, amount=1):
        self._concurrent_swap_wait_applied = ((self._concurrent_swap_wait_applied + amount) %
                                              self.amount_involved_nodes)

    def add_concurrent_sub_circuits(self, sub_circuits):
        if type(sub_circuits) != list:
            sub_circuits = [sub_circuits]
        for sub_circuit in sub_circuits:
            self._concurrent_sub_circuits.append(sub_circuit)

        self._concurrent_sub_circuits = list(set(self._concurrent_sub_circuits))

    def reset(self):
        self._total_duration = 0
        self._ran = False
        self._cut_off_time_reached = False
        self._concurrent_swap_wait_applied = 0

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        if type(other) != SubQuantumCircuit:
            raise ValueError("It is not possible to compare a SubQuantumCircuit object with anything else!")

        return self.name < other.name
