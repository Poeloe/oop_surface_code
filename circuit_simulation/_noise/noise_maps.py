import scipy.sparse as sp
import math
from circuit_simulation.termcolor.termcolor import colored
from circuit_simulation.gates.gates import *
from circuit_simulation.states.states import *
from circuit_simulation.basic_operations.basic_operations import CT
from circuit_simulation.gates.gate import SingleQubitGate


def N_amplitude_damping_channel(self, tqubit, density_matrix, num_qubits, waiting_time, T):
    p = 1 - math.exp(-waiting_time / T)
    kraus_opp_1 = SingleQubitGate("A1", np.array([[1, 0], [0, math.sqrt(1 - p)]]), 'A1')
    kraus_opp_2 = SingleQubitGate("A2", np.array([[0, math.sqrt(p)], [0, 0]]), 'A2')

    return N_kraus_operators(self, tqubit, [kraus_opp_1, kraus_opp_2], density_matrix, num_qubits)


def N_phase_damping_channel(self, tqubit, density_matrix, num_qubits, waiting_time, T):
    p = 1 - math.exp(-waiting_time / T)
    kraus_opp_1 = SingleQubitGate("K1", np.array([[1, 0], [0, math.sqrt(1 - p)]]), 'K1')
    kraus_opp_2 = SingleQubitGate("K2", np.array([[0, 0], [0, math.sqrt(p)]]), 'K2')

    return N_kraus_operators(self, tqubit, [kraus_opp_1, kraus_opp_2], density_matrix, num_qubits)


def N_combined_amplitude_phase_damping_channel(self, tqubit, density_matrix, num_qubits, waiting_time, T_a, T_p):
    """
        Obtained from https://quantumcomputing.stackexchange.com/questions/12857/find-the-kraus-operators-of-a-combined-
        amplitude-and-phase-damping-channel
    """
    p_a = 1 - math.exp(-waiting_time / T_a)
    p_p = 1 - math.exp(-waiting_time / T_p)

    kraus_opp_1 = SingleQubitGate("K1_ap", np.array([[1, 0], [0, math.sqrt(1 - p_a) * math.sqrt(1 - p_p)]]), 'K1_ap')
    kraus_opp_2 = SingleQubitGate("K2_ap", np.array([[0, math.sqrt(p_a)], [0, 0]]), 'K2_ap')
    kraus_opp_3 = SingleQubitGate("K3_ap", np.array([[0, 0], [0, math.sqrt(1 - p_a) * math.sqrt(p_p)]]), 'K3_ap')

    return N_kraus_operators(self, tqubit, [kraus_opp_1, kraus_opp_2, kraus_opp_3], density_matrix, num_qubits)


def N_kraus_operators(self, tqubit, kraus_operators, density_matrix, num_qubits):
    result = []
    for kraus_op in kraus_operators:
        kraus_op_full = self._create_1_qubit_gate(kraus_op, tqubit, num_qubits=num_qubits)
        result.append(kraus_op_full * CT(density_matrix, kraus_op_full))

    return sum(result)


def N_dephasing_channel(self, tqubit, density_matrix, num_qubits, p):
    Z_gate_full = self._create_1_qubit_gate(Z_gate, tqubit, num_qubits=num_qubits)

    return (1 - p) * density_matrix + p * (Z_gate_full * density_matrix * Z_gate_full)


def N_depolarising_channel(self, pg, tqubit, density_matrix, num_qubits, times=1, SWAP=False):
    """
        Private method to apply noise to the single qubit gates. This is done according to the equation

            N(rho) = (1-pg) * rho + pg/3 SUM_A [A * rho * A^], --> A in {X, Y, Z}

        in which '#' is the Kronecker product and ^ is the dagger (Hermitian conjugate).

        Parameters
        ----------
        pg : float [0-1]
            Indicates the amount of gate noise applied
        tqubit: int
            Integer that indicates the target qubit. Note that the qubit counting starts at 0.
        density_matrix : csr_matrix
            Density matrix to which the noise should be applied to.
        num_qubits : int
            Number of qubits of which the density matrix is composed.
        times : int
            Amount of application of the noise to the density matrix.
        SWAP : bool
            If used for the efficient swap gate, then an additional pre factor is necessary (two qubit gate error)
    """
    factor = 1 if not SWAP else 4/5
    x_full = self._create_1_qubit_gate(X_gate, tqubit, num_qubits=num_qubits)
    z_full = self._create_1_qubit_gate(Z_gate, tqubit, num_qubits=num_qubits)
    y_full = self._create_1_qubit_gate(Y_gate, tqubit, num_qubits=num_qubits)
    gates = [x_full, z_full, y_full]

    for _ in range(times):
        summed_matrix = sp.csr_matrix(density_matrix.shape)
        for gate in gates:
            # No CT used (so no 'A * CT(rho, A)' for speed-up), since X, Y and Z gates are symmetric
            summed_matrix += gate * (density_matrix * gate)
        density_matrix = (1-factor*pg) * density_matrix + factor*(pg/3) * summed_matrix
    return density_matrix


def N_two_qubit_gate(self, pg, cqubit, tqubit, density_matrix, num_qubits, times=1):
    """
        Private method to apply noise to the single qubit gates. This is done according to the equation

            N(rho) = (1-pg)*rho + pg/15 SUM_A SUM_B [(A # B) rho (A # B)^], --> {A, B} in {X, Y, Z, I}

        in which '#' is the Kronecker product and ^ is the dagger (Hermitian conjugate).

        Parameters
        ----------
        self: QuantumCircuit
            QuantumCircuit object
        pg : float [0-1]
            Indicates the amount of gate noise applied
        cqubit: int
            Integer that indicates the control qubit. Note that the qubit counting starts at 0.
        tqubit: int
            Integer that indicates the target qubit. Note that the qubit counting starts at 0.
        density_matrix : csr_matrix
            Density matrix to which the noise should be applied to.
        num_qubits : int
            Number of qubits of which the density matrix is composed.
        times : int
            Amount of times the error should be applied to the density matrix
    """
    for _ in range(times):
        density_matrix = ((1 - pg) * density_matrix + (pg / 15) * _double_sum_pauli_error(self,
                                                                                          cqubit,
                                                                                          tqubit,
                                                                                          density_matrix,
                                                                                          num_qubits=num_qubits))
    return density_matrix


def N_network(density_matrix, pn, network_noise_type):
    """
        Parameters
        ----------
        density_matrix : sparse matrix
            Density matrix of the ideal Bell-pair.
        pn : float [0-1]
            Amount of network noise present in the system.
        network_noise_type: int {0, 1}
            Type of network noise that is requested
            Option 0:
                (1-4/3*pn) * |Bell_ideal><Bell_ideal| + pn/3 * I
            Option 1:
                (1-pn) * |Bell_ideal><Bell_ideal| + pn * |11><11|
    """
    if network_noise_type not in [0, 1]:
        raise ValueError("Network noise type can only be 0 or 1, not {}".format(network_noise_type))

    if network_noise_type == 0:
        return (1-(4/3)*pn) * density_matrix + pn/3 * sp.eye(4, 4, format='csr')
    else:
        error_density = sp.lil_matrix((4, 4))
        error_density[3, 3] = 1
        return (1-pn) * density_matrix + pn * error_density


def N_preparation(state, p_prep):
    opp_state = state
    if state == ket_0:
        opp_state = ket_1
    if state == ket_1:
        opp_state = ket_0
    if state == ket_p:
        opp_state = ket_m
    if state == ket_m:
        opp_state = ket_p

    error_state = State("Prep error state",
                        (1-p_prep) * state.vector + p_prep * opp_state.vector,
                        colored("~", 'red') + state.representation)

    return error_state


def _sum_pauli_error_single(qc, tqubit, density_matrix, num_qubits):
    """
        Private method that calculates the pauli gate sum part of the equation specified in _N_single
        method, namely

            SUM_A [A * rho * A^], --> A in {X, Y, Z}

        Parameters
        ----------
        tqubit: int
            Integer that indicates the target qubit. Note that the qubit counting starts at 0.
        density_matrix : csr_matrix
            Density matrix to which the noise should be applied to.
        num_qubits : int
            Number of qubits of which the density matrix is composed.

        Returns
        -------
        summed_matrix : sparse matrix
            Returns a sparse matrix which is the result of the equation mentioned above.
    """

    gates = [X_gate, Y_gate, Z_gate]
    summed_matrix = sp.csr_matrix((2**num_qubits, 2**num_qubits))

    for gate in gates:
        pauli_error = qc._create_1_qubit_gate(gate, tqubit, num_qubits=num_qubits)
        summed_matrix = summed_matrix + pauli_error.dot(CT(density_matrix, pauli_error))
    return summed_matrix


def _double_sum_pauli_error(qc, qubit1, qubit2, density_matrix, num_qubits):
    """
        Private method that calculates the double pauli matrices sum part of the equation specified in _N
        method, namely

            SUM_B SUM_A [(A # B) * rho * (A # B)^], --> {A, B} in {X, Y, Z, I}

        in which '#' is the Kronecker product and ^ is the dagger (Hermitian conjugate).

        Parameters
        ----------
        qubit1: int
            Integer that indicates the either the target qubit or the control qubit. Note that the qubit counting
            starts at 0.
        qubit2 : int
            Integer that indicates the either the target qubit or the control qubit. Note that the qubit counting
            starts at 0.
        density_matrix : csr_matrix
            Density matrix to which the noise should be applied to.
        num_qubits : int
            Number of qubits of which the density matrix is composed.

        Returns
        -------
        summed_matrix : sparse matrix
            Returns a sparse matrix which is the result of the equation mentioned above.
    """
    gates = [X_gate, Y_gate, Z_gate, I_gate]
    qubit2_matrices = []

    result = sp.csr_matrix(density_matrix.shape)
    for i, gate_1 in enumerate(gates):
        # Create the full system 1-qubit gate for qubit1
        A = qc._create_1_qubit_gate(gate_1, qubit1, num_qubits=num_qubits)
        for j, gate_2 in enumerate(gates):
            # Create full system 1-qubit gate for qubit2, only once for every gate
            if i == 0:
                qubit2_matrices.append(qc._create_1_qubit_gate(gate_2, qubit2, num_qubits=num_qubits))

            # Skip the I*I case
            if i == j == len(gates) - 1:
                continue

            B = qubit2_matrices[j]
            result = result + (A * B).dot(CT(density_matrix, (A * B)))

    return sp.csr_matrix(result)