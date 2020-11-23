from circuit_simulation.stabilizer_measurement_protocols.run_protocols import _combine_superoperator_dataframes
import pandas as pd
import os
import re
import numpy as np


def get_all_files_from_folder(folder, folder_name, file_name):
    pattern = re.compile('^{}.*'.format(folder_name))
    files = []
    for sub_dir in os.listdir(folder):
        if pattern.fullmatch(sub_dir):
            for file in os.listdir(os.path.join(folder, sub_dir)):
                if file.endswith(".csv") and file_name == file:
                    files.append(os.path.join(folder, sub_dir, file))

    return files


def get_results_from_files(superoperator_files, name_csv):
    indices = ['protocol', 'pg', 'pm', 'pn']
    result_df = pd.DataFrame(columns=indices)
    result_df = result_df.set_index(indices)
    indices.pop(0)
    for protocol, superoperator_file in superoperator_files.items():
        result_df = result_df.sort_index()
        df = pd.read_csv(superoperator_file, sep=';', index_col=[0, 1])
        index = tuple(df.iloc[0, df.columns.get_loc(index)] for index in indices)
        index_list = list(index)
        index_list.insert(0, protocol)
        index = tuple(index_list)

        variables = ['written_to', 'avg_lde', 'avg_duration', 'ghz_fidelity']
        for variable in variables:
            result_df.loc[index, variable] = df.iloc[0, df.columns.get_loc(variable)]

        result_df.loc[index, 'IIII'] = df['p'].loc[('IIII', False)]
        result_df.loc[index, 'IIZZ'] = df['p'].loc[('IIZZ', False)]

    result_df.to_csv(name_csv, sep=';')


if __name__ == '__main__':
    name_csv = "./results/circuit_data.csv"
    folder = "./results/sim_data"
    folder_name = os.path.join(folder, "superoperator_prb_dec_swap")
    folder_name_2 = os.path.join(folder, "superoperator_prb_dec_no_swap")
    dict = {
        "plain_pulse_swap": "{}/pulse__plain_swap_pg0.001_pn0.05_pm0.01_pm_10.05_lde5000.csv"
            .format(folder_name),
        "expedient_pulse_swap": "{}/pulse__expedient_swap_pg0.001_pn0.05_pm0.01_pm_10.05_lde5000.csv"
            .format(folder_name),
        "plain_no_pulse_swap": "{}/no_pulse_plain_swap_pg0.001_pn0.05_pm0.01_pm_10.05_lde500.csv"
            .format(folder_name),
        "expedient_no_pulse_swap": "{}/no_pulse_expedient_swap_pg0.001_pn0.05_pm0.01_pm_10.05_lde500.csv"
            .format(folder_name),
        "expedient_no_pulse_no_swap": "{}/no_pulse_2_expedient_pg0.001_pn0.05_pm0.01_pm_10.05_lde500.csv"
            .format(folder_name_2),
        "plain_no_pulse_no_swap": "{}/no_pulse_2_plain_pg0.001_pn0.05_pm0.01_pm_10.05_lde500.csv"
            .format(folder_name_2),

    }

    get_results_from_files(dict, name_csv)
