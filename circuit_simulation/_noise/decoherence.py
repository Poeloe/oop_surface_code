

def N_decoherence(self, qubits=None, sub_circuit=None, sub_circuit_concurrent=False, decoherence=True):
    if not decoherence:
        return

    qubits = qubits if qubits is not None else [i for i in self.qubits.keys()]
    for qubit in qubits:
        current_qubit = self.qubits[qubit]
        waiting_time_lde = current_qubit.waiting_time_lde
        waiting_time_idle = current_qubit.waiting_time_idle
        if waiting_time_lde == waiting_time_idle == 0:
            continue
        qubit_type = current_qubit.qubit_type
        T2_idle = current_qubit.T2_idle
        T1_lde = current_qubit.T1_lde
        T2_lde = current_qubit.T2_lde
        density_matrix, qubits_dens, rel_qubit, rel_num_qubits = self._get_qubit_relative_objects(qubit)

        if waiting_time_idle > 0:
            density_matrix = self._N_phase_damping_channel(rel_qubit, density_matrix, rel_num_qubits, waiting_time_idle,
                                                           T2_idle)
            self._add_draw_operation("{:1.1e}xD[{}-{}]".format(waiting_time_idle, 'i', qubit_type), qubit, noise=True,
                                     sub_circuit=sub_circuit, sub_circuit_concurrent=sub_circuit_concurrent)
        if waiting_time_lde > 0:
            # Nuclear qubits will only actually experience the T1_lde and T2_lde. This is namely the decoherence time
            # for the qubits in the nodes not participating in the lde attempt
            if qubit_type == 'n':
                density_matrix = self._N_amplitude_damping_channel(rel_qubit, density_matrix, rel_num_qubits,
                                                                   waiting_time_lde, T1_lde)
                density_matrix = self._N_phase_damping_channel(rel_qubit, density_matrix, rel_num_qubits,
                                                               waiting_time_lde, T2_lde)
            self._add_draw_operation("{:1.1e}xD[{}-{}]".format(waiting_time_lde, 'l', qubit_type), qubit, noise=True,
                                     sub_circuit=sub_circuit, sub_circuit_concurrent=sub_circuit_concurrent)

        self._set_density_matrix(qubit, density_matrix)
        # After everything, set qubit waiting time to 0 again
        current_qubit.reset_waiting_time()
