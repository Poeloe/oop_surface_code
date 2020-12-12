from circuit_simulation.circuit_simulator import *


def cnot(qc: QuantumCircuit):
    qc.define_node("A", qubits=[0, 1])
    qc.define_node("B", qubits=[2, 3])
    qc.define_sub_circuit("AB")

    qc.start_sub_circuit("AB")
    qc.set_qubit_states({0: ket_1})
    qc.create_bell_pair(1, 2)
    qc.CNOT(0, 1)
    qc.CNOT(2, 3)
    outcome_b = qc.measure(2, basis="X")[0]
    outcome_a = qc.measure(1, basis="Z")[0]

    if outcome_a == 1:
        qc.X(3)
    if outcome_b == 1:
        qc.Z(0)

    qc.end_current_sub_circuit(total=True)

    return [0, 3]


def cnot_swap(qc: QuantumCircuit):
    qc.define_node("A", qubits=[0, 1], electron_qubits=1)
    qc.define_node("B", qubits=[2, 3], electron_qubits=2)
    qc.define_sub_circuit("AB")

    qc.start_sub_circuit("AB")
    qc.set_qubit_states({0: ket_1})
    qc.create_bell_pair(1, 2)
    qc.apply_gate(CNOT_gate, tqubit=1, cqubit=0, electron_is_target=True, reverse=True)
    qc.CNOT(2, 3)
    outcome_b = qc.measure(2, basis="X")[0]
    outcome_a = qc.measure(1, basis="Z")[0]

    qc.start_sub_circuit("AB")
    if outcome_a == 1:
        qc.X(3)
    if outcome_b == 1:
        qc.Z(0)

    qc.end_current_sub_circuit(total=True)

    return [0, 3]
