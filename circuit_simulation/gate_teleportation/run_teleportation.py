import circuit_simulation.gate_teleportation.teleportation_circuits as tel_circuits
from circuit_simulation.circuit_simulator import QuantumCircuit
from circuit_simulation.basic_operations.basic_operations import *
from circuit_simulation.states.states import *
from circuit_simulation.gate_teleportation.argument_parsing import compose_parser
from circuit_simulation.stabilizer_measurement_protocols.run_protocols import _additional_parsing_of_arguments, \
    _additional_qc_arguments
from itertools import product
import pandas as pd
from copy import copy
from tqdm import tqdm


def create_data_frame(data_frame, **kwargs):

    pop_list = ['iterations', 'save_latex_pdf', 'cp_path', 'color', 'draw_circuit', 'pb', 'two_qubit_gate_lookup',
                'single_qubit_gate_lookup', 'thread_safe_printing']
    index_columns = copy(kwargs)
    [index_columns.pop(item) for item in pop_list]

    if data_frame is not None:
        return data_frame, index_columns

    index = pd.MultiIndex.from_product([[item] for item in index_columns.values()], names=list(index_columns.keys()))
    data_frame = pd.DataFrame(index=index)
    data_frame['avg_fidelity'] = None

    return data_frame, index_columns


def run_series(iterations, gate, use_swap_gates, draw_circuit, color, pb, save_latex_pdf, cp_path, **kwargs):
    pbar = tqdm(total=iterations, position=1) if pb else None
    qc = QuantumCircuit(4, 0, **kwargs)
    gate = gate if not use_swap_gates else gate + '_swap'
    total_print_lines = []
    fidelities = []
    for i in range(iterations):
        pbar.update(1) if pb else None
        fid, print_lines = run_gate_teleportation(qc, gate, draw_circuit, color, **kwargs)
        total_print_lines.extend(print_lines)
        fidelities.append(fid)

    print(*total_print_lines)
    print("Average fideility is: {}".format(sum(fidelities)/iterations))
    return fidelities


def run_gate_teleportation(qc: QuantumCircuit, gate, draw_circuit, color, **kwargs):
    teleportation_circuit = getattr(tel_circuits, gate)
    qubits = teleportation_circuit(qc)
    fid = qc.get_state_fidelity(qubits, CT(ket_1 * ket_1), set_ghz_fidelity=False)

    if draw_circuit:
        qc.draw_circuit(no_color=not color, color_nodes=True)
        qc.append_print_lines("\nFidelity: {}".format(fid))

    print_lines = qc.print_lines
    qc.reset()

    return fid, print_lines


def run_for_arguments(gates, gate_error_probabilities, network_error_probabilities, meas_error_probabilities,
                      meas_error_probabilities_one_state, csv_filename, pm_equals_pg,
                      fixed_lde_attempts, threaded, **kwargs):

    meas_1_errors = [None] if meas_error_probabilities_one_state is None else meas_error_probabilities_one_state
    meas_errors = [None] if meas_error_probabilities is None else meas_error_probabilities
    pb = kwargs.pop('no_progress_bar')
    data_frame = None
    pbar1 = tqdm(total=len(list(product(gates, gate_error_probabilities, network_error_probabilities,
                                               meas_errors, meas_1_errors, fixed_lde_attempts))), position=0)

    # Loop over command line arguments
    for gate, pg, pn, pm, pm_1, lde in product(gates, gate_error_probabilities, network_error_probabilities,
                                               meas_errors, meas_1_errors, fixed_lde_attempts):
        pbar1.update(1)
        pm = pg if pm is None or pm_equals_pg else pm
        loop_arguments = {
            'gate': gate,
            'pg': pg,
            'pm': pm,
            'pn': pn,
            'pm_1': pm_1,
            'fixed_lde_attempts': lde,
            'pb': pb
        }
        kwargs.update(loop_arguments)
        kwargs = _additional_qc_arguments(**kwargs)
        data_frame, index_columns = create_data_frame(data_frame, **kwargs)
        if threaded:
            pass
        else:
            fidelities = run_series(**kwargs)
            data_frame.loc[tuple(index_columns.values()), 'avg_fidelity'] = sum(fidelities)/kwargs['iterations']
    data_frame.to_csv(csv_filename + '.csv')


if __name__ == '__main__':
    parser = compose_parser()

    args = vars(parser.parse_args())
    args = _additional_parsing_of_arguments(args)
    args.pop('gate_duration_file')

    run_for_arguments(**args)
