import unittest
from circuit_simulation.circuit_simulator import QuantumCircuit as QC
from circuit_simulation.basic_operations.basic_operations import *
from circuit_simulation.states.states import *
from circuit_simulation.gates.gates import *


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

    def test_fidelity_1(self):
        rho_0 = sp.lil_matrix([[1, 0], [0, 0]])
        fid = fidelity(rho_0, rho_0)
        self.assertEqual(fid, 1)

    def test_fidelity_0(self):
        rho_0 = sp.lil_matrix([[1, 0], [0, 0]])
        rho_1 = sp.lil_matrix([[0, 0], [0, 1]])
        self.assertEqual(fidelity(rho_0, rho_1), 0)
        self.assertEqual(fidelity(rho_1, rho_0), 0)

    def test_fidelity_half(self):
        rho_p = sp.lil_matrix([[1/2, 1/2], [1/2, 1/2]])
        rho_0 = sp.lil_matrix([[1, 0], [0, 0]])
        self.assertAlmostEqual(fidelity(rho_p, rho_0), 1/2)
        self.assertAlmostEqual(fidelity(rho_0, rho_p), 1 / 2)

    def test_fidelity_elementwise_half(self):
        rho_p = sp.lil_matrix([[1/2, 1/2], [1/2, 1/2]])
        rho_0 = sp.lil_matrix([[1, 0], [0, 0]])
        self.assertAlmostEqual(fidelity_elementwise(rho_0, rho_p), 1/2)
        self.assertAlmostEqual(fidelity_elementwise(rho_p, rho_0), 1/2)


class TestQuantumCircuitInit(unittest.TestCase):

    def test_basic_init(self):
        qc = QC(4, 0)
        self.assertEqual(qc.num_qubits, 4)
        self.assertEqual(qc.d, 2**4)
        self.assertEqual(qc.total_density_matrix()[0].shape, (2**4, 2**4))

    def test_first_qubit_ket_p_init(self):
        qc = QC(2, 1)
        density_matrix = np.array([[1/2, 0, 1/2, 0], [0, 0, 0, 0], [1/2, 0, 1/2, 0], [0, 0, 0, 0]])

        self.assertEqual(qc.num_qubits, 2)
        self.assertEqual(qc.d, 2 ** 2)
        self.assertEqual(qc.total_density_matrix()[0].shape, (2 ** 2, 2 ** 2))
        self.assertEqual(qc._qubit_array[0], ket_p)
        np.testing.assert_array_almost_equal(qc.total_density_matrix()[0].toarray(), density_matrix)

    def test_bell_pair_init(self):
        qc = QC(8, 2)
        density_matrix = np.array([[1/2, 0, 0, 1/2], [0, 0, 0, 0], [0, 0, 0, 0], [1/2, 0, 0, 1/2]])

        matrix_01, _, _, _ = qc._get_qubit_relative_objects(0)
        matrix_23, _, _, _ = qc._get_qubit_relative_objects(2)
        matrix_45, _, _, _ = qc._get_qubit_relative_objects(4)
        matrix_67, _, _, _ = qc._get_qubit_relative_objects(6)

        self.assertEqual(qc.num_qubits, 8)
        self.assertEqual(qc.d, 2 ** 8)
        self.assertEqual(qc.total_density_matrix()[0].shape, (2 ** 8, 2 ** 8))
        np.testing.assert_array_equal(matrix_01.toarray(), density_matrix)
        np.testing.assert_array_equal(matrix_23.toarray(), density_matrix)
        np.testing.assert_array_equal(matrix_45.toarray(), density_matrix)
        np.testing.assert_array_equal(matrix_67.toarray(), density_matrix)


class TestQuantumCircuitGates(unittest.TestCase):

    def test_one_qubit_gate_X(self):
        qc = QC(2, 0)
        X_gate_test = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])

        gate_result = qc._create_1_qubit_gate(X_gate, 0)
        np.testing.assert_array_equal(gate_result.toarray(), X_gate_test)

    def test_one_qubit_gate_Z(self):
        qc = QC(2, 0)
        Z_gate_test = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])

        gate_result = qc._create_1_qubit_gate(Z_gate, 0)
        np.testing.assert_array_equal(gate_result.toarray(), Z_gate_test)

    def test_one_qubit_gate_Y(self):
        qc = QC(2, 0)
        Y_gate_test = np.array([[0, 0, -1j, 0], [0, 0, 0, -1j], [1j, 0, 0, 0], [0, 1j, 0, 0]])

        gate_result = qc._create_1_qubit_gate(Y_gate, 0)
        np.testing.assert_array_equal(gate_result.toarray(), Y_gate_test)

    def test_one_qubit_gate_H(self):
        qc = QC(2, 0)
        H_gate_test = (1/np.sqrt(2)) * np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, -1, 0], [0, 1, 0, -1]])

        gate_result = qc._create_1_qubit_gate(H_gate, 0)
        np.testing.assert_array_equal(gate_result.toarray(), H_gate_test)

    def test_CNOT_gate_cqubit_0(self):
        qc = QC(2, 0)
        CNOT_gate_test = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

        gate_result = qc._create_2_qubit_gate(CNOT_gate, 0, 1)
        np.testing.assert_array_equal(gate_result.toarray(), CNOT_gate_test)

    def test_CNOT_gate_cqubit_1(self):
        qc = QC(2, 0)
        CNOT_gate_test = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])

        gate_result = qc._create_2_qubit_gate(CNOT_gate, 1, 0)
        np.testing.assert_array_equal(gate_result.toarray(), CNOT_gate_test)

    def test_CZ_gate_cqubit_0(self):
        qc = QC(2, 0)
        CZ_gate_test = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

        gate_result = qc._create_2_qubit_gate(CZ_gate, 0, 1)
        np.testing.assert_array_equal(gate_result.toarray(), CZ_gate_test)

    def test_CZ_gate_cqubit_1(self):
        qc = QC(2, 0)
        CZ_gate_test = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

        gate_result = qc._create_2_qubit_gate(CZ_gate, 1, 0)
        np.testing.assert_array_equal(gate_result.toarray(), CZ_gate_test)

    def test_SWAP_gate(self):
        qc = QC(2, 0)

        gate_result = qc._create_2_qubit_gate(SWAP_gate, 1, 0)
        np.testing.assert_array_equal(gate_result.toarray(), SWAP_gate.matrix)

    def test_SWAP_gate_cqubit_1(self):
        qc = QC(2, 0)

        gate_result = qc._create_2_qubit_gate(SWAP_gate, 0, 1)
        np.testing.assert_array_equal(gate_result.toarray(), SWAP_gate.matrix)


class TestErrorImplementation(unittest.TestCase):

    def test_single_gate_error(self):
        qc = QC(1, 0, noise=True, pg=0.01)
        qc.X(0)

        expected_density_matrix = np.array([[2/3*0.01, 0], [0, (1-0.01)+0.01/3]])
        np.testing.assert_array_almost_equal(qc.total_density_matrix()[0].toarray().real, expected_density_matrix)

    def test_two_qubit_gate_error(self):
        qc = QC(2, 0, noise=True, pg=0.01)
        qc.CNOT(0, 1)

        expected_density_matrix = np.array([[(1-(0.01*12/15)), 0, 0, 0],
                                            [0, 0.04/15, 0, 0],
                                            [0, 0, 0.04/15, 0],
                                            [0, 0, 0, 0.04/15]])
        np.testing.assert_array_almost_equal(qc.total_density_matrix()[0].toarray().real, expected_density_matrix)

    def test_amplitude_damping_channel(self):
        qc = QC(1, 0)
        density_matrix = sp.csr_matrix([[0, 0], [0, 1]])
        compare_matrix = sp.csr_matrix([[1, 0], [0, 0]])

        density_matrix_noise = qc._N_amplitude_damping_channel(0, density_matrix, 1, 10, 2.3)
        fid = fidelity_elementwise(density_matrix, density_matrix_noise)

        np.testing.assert_array_almost_equal(compare_matrix.toarray(), density_matrix_noise.toarray(), 2)
        self.assertLess(fid, 0.1)

    def test_phase_damping_channel(self):
        qc = QC(1, 0)
        density_matrix = sp.csr_matrix([[1/2, 1/2], [1/2, 1/2]])
        compare_matrix = sp.csr_matrix([[1/2, 0], [0, 1/2]])

        density_matrix_noise = qc._N_phase_damping_channel(0, density_matrix, 1, 20, 2.3)
        fid = fidelity_elementwise(density_matrix, density_matrix_noise)

        np.testing.assert_array_almost_equal(compare_matrix.toarray(), density_matrix_noise.toarray(), 2)
        self.assertLess(fid, 0.6)

class TestMeasurement(unittest.TestCase):

    def test_measure_first_N_qubit(self):
        # Initialise system in |+0> state, CNOT on 2nd qubit and measure |+> on first qubit
        qc = QC(2, 1)
        qc.CNOT(0, 1)
        qc.measure_first_N_qubits(1, measure=0)

        correct_result = np.array([[0.5, 0.5], [0.5, 0.5]])
        np.testing.assert_array_equal(qc.total_density_matrix()[0].toarray(), correct_result)

        # Initialise second system also in |+0>, CNOT on 2nd qubit and measure |-> on first qubit
        qc2 = QC(2, 1)
        qc2.CNOT(0, 1)
        qc2.measure_first_N_qubits(1, measure=1)

        correct_result_2 = np.array([[0.5, -0.5], [-0.5, 0.5]])
        np.testing.assert_array_equal(qc2.total_density_matrix()[0].toarray(), correct_result_2)

    def test_measure_first_qubit_plus_x_basis(self):
        qc = QC(2, 1)
        qc.measure(0, outcome=0)

        correct_result = 1/2 * np.array([[1, 0, 1, 0], [0, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]])
        np.testing.assert_array_almost_equal(qc.total_density_matrix()[0].toarray().real, correct_result)

    def test_measure_first_qubit_minus_x_basis(self):
        qc = QC(2, 1)
        with self.assertRaises(ValueError) as error:
            qc.measure(0, outcome=1)
        self.assertTrue(error.exception)

    def test_measure_first_qubit_bell_state_plus(self):
        # Initialise system in |+0> state, CNOT on 2nd qubit and measure |+> on first qubit
        qc = QC(2, 1)
        qc.CNOT(0, 1)
        qc.measure(0, outcome=0)

        correct_result = 1/4 * np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
        np.testing.assert_array_almost_equal(qc.total_density_matrix()[0].toarray().real, correct_result)

    def test_measure_first_qubit_bell_state_minus(self):
        # Initialise second system also in |+0>, CNOT on 2nd qubit and measure |-> on first qubit
        qc = QC(2, 1)
        qc.CNOT(0, 1)
        qc.measure(0, outcome=1)

        correct_result = 1/4 * np.array([[1, -1, -1, 1], [-1, 1, 1, -1], [-1, 1, 1, -1], [1, -1, -1, 1]])
        np.testing.assert_array_almost_equal(qc.total_density_matrix()[0].toarray().real, correct_result)

    def test_measure_first_qubit_bell_state_zero(self):
        qc = QC(2, 1)
        qc.CNOT(0, 1)
        qc.measure(0, outcome=0, basis="Z")

        correct_result = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        np.testing.assert_array_almost_equal(qc.total_density_matrix()[0].toarray().real, correct_result)

    def test_measure_first_qubit_bell_state_one(self):
        qc = QC(2, 1)
        qc.CNOT(0, 1)
        qc.measure(0, outcome=1, basis="Z")

        correct_result = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        np.testing.assert_equal(qc.total_density_matrix()[0].toarray().real, correct_result)

    def test_outcome_probabilities(self):
        for _ in range(100):
            qc = QC(4, 0, probabilistic=True)
            qc.create_bell_pair(0, 1)
            outcome = qc.measure([1, 0])

            self.assertEqual(outcome[0], outcome[1])

    def test_outcome_probabilities_CZ(self):
        for _ in range(100):
            qc = QC(4, 0, probabilistic=True)
            qc.create_bell_pair(0, 1)
            qc.CZ(1, 0)
            outcome = qc.measure([1, 0])

            self.assertFalse(outcome[0] == outcome[1])

    def test_outcome_probabilities_cntrl(self):
        for _ in range(100):
            qc = QC(4, 0, probabilistic=True)
            qc.create_bell_pair(0, 1)
            qc.CZ(0, 2)
            outcome = qc.measure([1, 0])

            self.assertEqual(outcome[0], outcome[1])

    def test_outcome_probabilities_single_selection(self):
        for _ in range(100):
            qc = QC(4, 0, probabilistic=True)
            qc.create_bell_pair(3, 1)
            qc.create_bell_pair(2, 0)
            qc.CZ(0, 1)
            qc.CZ(2, 3)
            outcome = qc.measure([0, 2])

            self.assertEqual(outcome[0], outcome[1])

    def test_outcome_probabilities_single_dot(self):
        for _ in range(100):
            qc = QC(6, 0, probabilistic=True)
            qc.create_bell_pair(5, 2)
            qc.create_bell_pair(4, 1)
            qc.single_selection(CNOT_gate, 3, 0)
            qc.single_selection(CZ_gate, 3, 0)
            qc.CZ(4, 5)
            qc.CZ(1, 2)
            outcomes = qc.measure([1, 4])

            self.assertEqual(outcomes[0], outcomes[1])
            bell_matrix = 1/2 * np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]])
            np.testing.assert_array_equal(qc._qubit_density_matrix_lookup[5][0].toarray(), bell_matrix)


class TestSeparatedDensityMatrices(unittest.TestCase):

    def testDensityMatrixInit(self):
        qc = QC(10, 0)
        for qubit, (density_matrix, qubits) in qc._qubit_density_matrix_lookup.items():
            np.testing.assert_array_equal(sp.csr_matrix([[1, 0], [0, 0]]).toarray(), density_matrix.toarray())
            self.assertListEqual([qubit], qubits)

    def testTwoQubitGateFusion(self):
        qc = QC(10, 0)
        qc.apply_gate(CNOT_gate, cqubit=0, tqubit=1)

        density_matrix_0, qubits_0 = qc._qubit_density_matrix_lookup[0]
        density_matrix_1, qubits_1 = qc._qubit_density_matrix_lookup[1]
        self.assertTrue(density_matrix_0 is density_matrix_1)
        self.assertTrue(qubits_0 is qubits_1)
        # Qubits must be fused in the order control qubits + target qubits
        self.assertEqual(qubits_0, [0, 1])

    def testBellPairFusion(self):
        qc = QC(5, 0)
        qc.create_bell_pair(0, 1)

        density_matrix_0, qubits_0 = qc._qubit_density_matrix_lookup[0]
        density_matrix_1, qubits_1 = qc._qubit_density_matrix_lookup[1]
        self.assertTrue(density_matrix_0 is density_matrix_1)
        self.assertTrue(qubits_0 is qubits_1)
        # Second qubit of 'create_bell_pair' should be the first qubit in the density matrix
        self.assertEqual(qubits_0, [1, 0])

    def testSWAP(self):
        qc = QC(5, 0)
        qc.X(0)
        qc.SWAP(0, 1)

        density_matrix_0, qubits_0 = qc._qubit_density_matrix_lookup[0]
        density_matrix_1, qubits_1 = qc._qubit_density_matrix_lookup[1]
        np.testing.assert_array_equal(density_matrix_0.toarray(), np.array([[1, 0], [0, 0]]))
        np.testing.assert_array_equal(density_matrix_1.toarray(), np.array([[0, 0], [0, 1]]))
        self.assertEqual(qubits_0, [0])
        self.assertEqual(qubits_1, [1])

    def testSWAPBellPair(self):
        qc = QC(5, 0)
        qc.create_bell_pair(0, 1)
        qc.SWAP(0, 2)
        qc.SWAP(1, 3)

        density_matrix_0, qubits_0 = qc._qubit_density_matrix_lookup[0]
        density_matrix_1, qubits_1 = qc._qubit_density_matrix_lookup[1]
        density_matrix_2, qubits_2 = qc._qubit_density_matrix_lookup[2]
        density_matrix_3, qubits_3 = qc._qubit_density_matrix_lookup[3]
        self.assertTrue(density_matrix_2 is density_matrix_3)
        self.assertTrue(density_matrix_2.shape == (4, 4))
        self.assertTrue(qubits_2 is qubits_3)
        np.testing.assert_array_equal(density_matrix_0.toarray(), np.array([[1, 0], [0, 0]]))
        np.testing.assert_array_equal(density_matrix_1.toarray(), np.array([[1, 0], [0, 0]]))
        self.assertTrue(qubits_0 is not qubits_1)
        self.assertEqual(qubits_0, [0])
        self.assertEqual(qubits_1, [1])
        self.assertEqual(qubits_3, [3, 2])


if __name__ == '__main__':
    unittest.main()
