import pandas as pd
import os
import re


def get_all_files_from_folder(folder, folder_name):
    pattern = re.compile('^{}.*'.format(folder_name))
    files = []
    for sub_dir in os.listdir(folder):
        if pattern.fullmatch(sub_dir):
            for file in os.listdir(os.path.join(folder, sub_dir)):
                if file.endswith(".csv"):
                    files.append(os.path.join(folder, sub_dir, file))

    return files


def get_results_from_files(superoperator_files, name_csv):
    indices = ['protocol_name', 'pg', 'pm', 'pm_1', 'pn', 'decoherence', 'p_bell_success', 'pulse_duration',
               'network_noise_type', 'no_single_qubit_error', 'basis_transformation_noise', 'cut_off_time',
               'probabilistic', 'fixed_lde_attempts']
    result_df = pd.DataFrame(columns=indices)
    result_df = result_df.set_index(indices)
    for superoperator_file in superoperator_files:
        result_df = result_df.sort_index()
        df = pd.read_csv(superoperator_file, sep=';', index_col=[0, 1], float_precision='round_trip')
        if df.iloc[0, df.columns.get_loc('pulse_duration')] == 0:
            df.iloc[0, df.columns.get_loc('fixed_lde_attempts')] = 0
        index = tuple(df.iloc[0, df.columns.get_loc(index)] for index in indices)

        variables = ['written_to', 'total_lde_attempts', 'avg_lde_attempts', 'total_duration', 'avg_duration',
                     'ghz_fidelity', 'int_dur', 'int_ghz', 'int_stab']
        for variable in variables:
            if variable in result_df:
                result_df.loc[index, variable] = df.iloc[0, df.columns.get_loc(variable)]

        result_df.loc[index, 'IIII'] = df['p'].loc[('IIII', False)]
        result_df.loc[index, 'IIZZ'] = df['p'].loc[('IIZZ', False)]

    result_df = result_df.sort_index()
    result_df.to_csv(name_csv, sep=';')
    print(result_df)


if __name__ == '__main__':
    name_csv = "./results/circuit_data_NV.csv"
    folder = "./results/sim_data_3"
    folder_name = "superoperator_fid_dur"

    # files = [
    #     "{}/no_pulse_dyn_prot_4_4_1_swap_pg0.01_pn0.05_pm0.01_pm_10.05_lde5000.csv".format(folder_name),
    #     "{}/no_pulse_expedient_swap_pg0.01_pn0.05_pm0.01_pm_10.05_lde5000.csv".format(folder_name),
    #     "{}/no_pulse_plain_swap_pg0.01_pn0.05_pm0.01_pm_10.05_lde5000.csv".format(folder_name),
    # ]

    files = get_all_files_from_folder(folder, folder_name)

    get_results_from_files(files, name_csv)
