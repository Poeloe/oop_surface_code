import unittest
from circuit_simulation.circuit_simulator import QuantumCircuit as QC
import numpy as np
import scipy.sparse as sp
from circuit_simulation.basic_operations import (KP, CT, fidelity, trace, ket_0, ket_1, ket_p, X, Z, Y, H)


class TestBasicOperations(unittest.TestCase):

    def test_kronecker_product(self):
        ket_01 = sp.lil_matrix([[0, 1, 0, 0]]).T
        product = KP(ket_0, ket_1)
        np.testing.assert_array_equal(ket_01.toarray(), product.toarray())

    def test_CT(self):
        rho_01 = sp.lil_matrix([[0, 1], [0, 0]])
        CT_product = CT(ket_0, ket_1)
        np.testing.assert_array_equal(rho_01.toarray(), CT_product.toarray())

    def test_trace(self):
        id_2 = sp.eye(4, 4)
        self.assertEqual(trace(id_2), 4)

    def test_fidelity(self):
        rho_0 = sp.lil_matrix([[1, 0], [0, 0]])
        fid = fidelity(rho_0, rho_0)
        self.assertEqual(fid, 1)


class TestQuantumCircuitInit(unittest.TestCase):

    def test_basic_init(self):
        qc = QC(4, 0)
        self.assertEqual(qc.num_qubits, 4)
        self.assertEqual(qc.d, 2**4)
        self.assertEqual(qc.density_matrix.shape, (2**4, 2**4))

    def test_first_qubit_ket_p_init(self):
        qc = QC(2, 1)
        density_matrix = np.array([[1/2, 0, 1/2, 0], [0, 0, 0, 0], [1/2, 0, 1/2, 0], [0, 0, 0, 0]])

        self.assertEqual(qc.num_qubits, 2)
        self.assertEqual(qc.d, 2 ** 2)
        self.assertEqual(qc.density_matrix.shape, (2 ** 2, 2 ** 2))
        np.testing.assert_array_equal(qc.get_begin_states()[0], ket_p)
        np.testing.assert_array_equal(qc.density_matrix.toarray(), density_matrix)

    def test_bell_pair_init(self):
        qc = QC(2, 2)
        density_matrix = np.array([[1/2, 0, 0, 1/2], [0, 0, 0, 0], [0, 0, 0, 0], [1/2, 0, 0, 1/2]])

        self.assertEqual(qc.num_qubits, 2)
        self.assertEqual(qc.d, 2 ** 2)
        self.assertEqual(qc.density_matrix.shape, (2 ** 2, 2 ** 2))
        np.testing.assert_array_equal(qc.density_matrix.toarray(), density_matrix)


class TestQuantumCircuitGates(unittest.TestCase):

    def test_one_qubit_gate_X(self):
        qc = QC(2, 0)
        X_gate = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])

        gatee_result = qc._create_1_qubit_gate(X, 0)
        np.testing.assert_array_equal(gatee_result.toarray(), X_gate)

    def test_one_qubit_gate_Z(self):
        qc = QC(2, 0)
        Z_gate = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])

        gate_result = qc._create_1_qubit_gate(Z, 0)
        np.testing.assert_array_equal(gate_result.toarray(), Z_gate)

    def test_one_qubit_gate_Y(self):
        qc = QC(2, 0)
        Y_gate = np.array([[0, 0, -1j, 0], [0, 0, 0, -1j], [1j, 0, 0, 0], [0, 1j, 0, 0]])

        gate_result = qc._create_1_qubit_gate(Y, 0)
        np.testing.assert_array_equal(gate_result.toarray(), Y_gate)

    def test_one_qubit_gate_H(self):
        qc = QC(2, 0)
        H_gate = (1/np.sqrt(2)) * np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, -1, 0], [0, 1, 0, -1]])

        gate_result = qc._create_1_qubit_gate(H, 0)
        np.testing.assert_array_equal(gate_result.toarray(), H_gate)

    def test_CNOT_gate_cqubit_0(self):
        qc = QC(2, 0)
        CNOT_gate = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

        gate_result = qc._create_2_qubit_gate(X, 0, 1)
        np.testing.assert_array_equal(gate_result.toarray(), CNOT_gate)

    def test_CNOT_gate_cqubit_1(self):
        qc = QC(2, 0)
        CNOT_gate = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])

        gate_result = qc._create_2_qubit_gate(X, 1, 0)
        np.testing.assert_array_equal(gate_result.toarray(), CNOT_gate)

    def test_CZ_gate_cqubit_0(self):
        qc = QC(2, 0)
        CZ_gate = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

        gate_result = qc._create_2_qubit_gate(Z, 0, 1)
        np.testing.assert_array_equal(gate_result.toarray(), CZ_gate)

    def test_CZ_gate_cqubit_1(self):
        qc = QC(2, 0)
        CZ_gate = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

        gate_result = qc._create_2_qubit_gate(Z, 1, 0)
        np.testing.assert_array_equal(gate_result.toarray(), CZ_gate)


if __name__ == '__main__':
    unittest.main()