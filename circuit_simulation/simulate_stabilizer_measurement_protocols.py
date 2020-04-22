from qiskit import execute, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer
from qiskit.quantum_info import Operator
from qiskit.compiler import assemble
import qiskit.providers.aer.noise as noise
from qiskit.providers.aer import QasmSimulator
from pprint import pprint


noisy_bell = Operator([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

def double_selection(qc, qubit1, qubit2, gate_type="z"):
    create_bell_pair(qc, qubit1, qubit2)

    if gate_type == "z":
        qc.cz(qubit1, qubit1 + 1)
        qc.cz(qubit2, qubit2 + 1)
    else:
        qc.cx(qubit1, qubit1 + 1)
        qc.cx(qubit2, qubit2 + 1)

    create_bell_pair(qc, qubit1 - 1, qubit2 - 1)
    qc.cz(qubit1 - 1, qubit1)
    qc.cz(qubit2 - 1, qubit2)

    x_measurement(qc, qubit1 - 1, qubit1 - 1)
    x_measurement(qc, qubit1, qubit1)
    x_measurement(qc, qubit2 - 1, qubit2 - 1)
    x_measurement(qc, qubit2, qubit2)

def extended_double_selection(qc, qubit1, qubit2, gate_type="z", protocol_type="expedient"):
    create_bell_pair(qc, qubit1, qubit2)

    create_bell_pair(qc, qubit1 - 1, qubit2 - 1)
    qc.cx(qubit1-1, qubit1)
    qc.cx(qubit2-1, qubit2)
    x_measurement(qc, qubit1 - 1, qubit1 - 1)
    x_measurement(qc, qubit2 - 1, qubit2 - 1)

    create_bell_pair(qc, qubit1 - 1, qubit2 - 1)
    qc.cz(qubit1-1, qubit1)
    qc.cz(qubit2-1, qubit2)
    x_measurement(qc, qubit1 - 1, qubit1 - 1)
    x_measurement(qc, qubit2 - 1, qubit2 - 1)

    if gate_type == "z":
        qc.cz(qubit1, qubit1 + 1)
        qc.cz(qubit2, qubit2 + 1)
    else:
        qc.cx(qubit1, qubit1 + 1)
        qc.cx(qubit2, qubit2 + 1)

    if protocol_type == "stringent":
        create_bell_pair(qc, qubit1 - 1, qubit2 - 1)
        qc.rz(qubit1 - 1, qubit1)
        qc.rz(qubit2 - 1, qubit2)
        x_measurement(qc, qubit1 - 1, qubit1 - 1)
        x_measurement(qc, qubit2 - 1, qubit2 - 1)

    x_measurement(qc, qubit1, qubit1)
    x_measurement(qc, qubit2, qubit2)


def create_bell_pair(qc, q1, q2, noisy=True):
    qc.h(q1)
    qc.cx(q1, q2)
    if noisy:
        qc.unitary(noisy_bell, [q1, q2], label="bell")


def stabilize(qc, qubit, type):
    if type == "x":
        qc.cx(qubit - 1, qubit)
    else:
        qc.cz(qubit - 1, qubit)
    x_measurement(qc, qubit - 1, qubit - 1)


def x_measurement(qc, qubit, cbit):
    """Measure 'qubit' in the X-basis, and store the result in 'cbit'"""
    qc.h(qubit)
    qc.measure(qubit, cbit)
    qc.h(qubit)
    return qc


def GHZ_stabilizer(stab_type, protocol_type, draw=False):
    # Create needed qubits for GHZ stabilizer
    qr = QuantumRegister(8)
    cr = ClassicalRegister(8)
    qc = QuantumCircuit(qr, cr)

    for i in range(1):
        # Phase 1: Create distilled Bell pairs
        create_bell_pair(qc, 2 + i*8, 6 + i*8)
        double_selection(qc, 1 + i*8, 5 + i*8)
        double_selection(qc, 1 + i*8, 5 + i*8, "x")
        if protocol_type == "stringent":
            extended_double_selection(qc, 1 + i*8, 5 + i*8, protocol_type=protocol_type)
            extended_double_selection(qc, 1 + i*8, 5 + i*8, "x", protocol_type)

    # for j in range(2):
    #     # Phase 2: Create GHZ states
    #     extended_double_selection(qc, 1 + j*4, 9 + j*4, protocol_type=protocol_type)
    #     extended_double_selection(qc, 1 + j*4, 9 + j*4, protocol_type=protocol_type)

    stabilize(qc, 3, stab_type)
    stabilize(qc, 7, stab_type)
    # stabilize(qc, 11, stab_type)
    # stabilize(qc, 15, stab_type)

    if draw:
        qc.draw(output="mpl", filename="/Users/Paul/Desktop/circuit.png")

    return qc


def create_noise_model(pn, pg, pm):
    network_error = noise.depolarizing_error(pn, 2)
    gate_error = noise.depolarizing_error(pg, 2)
    meas_error = noise.pauli_error([('X', pm), ('I', 1 - pm)])

    noise_model = noise.NoiseModel()
    noise_model.add_all_qubit_quantum_error(network_error, 'bell')
    noise_model.add_all_qubit_quantum_error(gate_error, ['cz', 'cx'])
    noise_model.add_all_qubit_quantum_error(meas_error, "measure")
    noise_model.add_basis_gates(['unitary'])

    return noise_model


pn = 0.1
pg = 0.006
pm = 0.006

noise_model = create_noise_model(pn, pg, pm)
circuit = GHZ_stabilizer("x", "expedient")
backend = Aer.get_backend('statevector_simulator')

job = execute(circuit, backend)

result = job.result()

print(result.get_data())
