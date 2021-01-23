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
    qc.define_node("A", qubits=[4, 1])
    qc.define_node("B", qubits=[2, 0])
    qc.define_sub_circuit("AB")

    qc.start_sub_circuit("AB")
    qc.create_bell_pair("A-e", "B-e")
    qc.apply_gate(CNOT_gate, tqubit="A-e", cqubit="A-0", electron_is_target=True, reverse=True)
    qc.CNOT("B-e", "B-0")
    outcome_b = qc.measure("B-e", basis="X")[0]
    outcome_a = qc.measure("A-e", basis="Z")[0]

    qc.start_sub_circuit("AB")
    if outcome_a == 1:
        qc.X("B-0")
    if outcome_b == 1:
        qc.Z("A-0")

    qc.end_current_sub_circuit(total=True, apply_decoherence=True, forced_level=True)

    return qc.get_combined_density_matrix([4, 5, 2, 3])[0]
