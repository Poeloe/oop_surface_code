import math


def N_decoherence_new(self, qubits, p_dec=None):
    for qubit in qubits:
        current_qubit = self.qubits[qubit]
        waiting_time_lde = current_qubit.waiting_time_lde
        waiting_time_idle = current_qubit.waiting_time_idle
        qubit_type = current_qubit.qubit_type
        density_matrix, qubits_dens, rel_qubit, rel_num_qubits = self._get_qubit_relative_objects(qubit)

        if waiting_time_idle > 0:
            # TODO: create an 'a' value that is specific for the idle case and distinct for the electron qubit.
            #  (User should be able to adapt this parameter)
            a = 0.04 if qubit_type == 'e' else 0.004
            p_dec = (3 - 3 * math.exp(-a * waiting_time_idle) ) /4 if not p_dec else p_dec
            self._N_depolarising_channel(p_dec, rel_qubit, density_matrix, rel_num_qubits)
            self._add_draw_operation("{:.2}xD[{}]".format(waiting_time_idle, 'idle'), qubit, noise=True)
        if waiting_time_lde > 0:
            # TODO: create an 'a' value that corresponds to the idle + LDE case. In the case of LDE also T1 should
            #  be taken into account (should be adaptable by the user)
            a = 0.4
            p_dec = (3 - 3 * math.exp(-a * waiting_time_lde)) / 4 if not p_dec else p_dec
            self._N_depolarising_channel(p_dec, rel_qubit, density_matrix, rel_num_qubits)
            self._add_draw_operation("{:.2}xD[{}]".format(waiting_time_lde, 'lde'), qubit, noise=True)

        # After everything, set qubit waiting time to 0 again
        current_qubit.reset_waiting_time()
