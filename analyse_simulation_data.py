from circuit_simulation.stabilizer_measurement_protocols.run_protocols import _combine_superoperator_dataframes
import pandas as pd
import os
import re
import numpy as np


def get_all_files_from_folder(folder, folder_name):
    pattern = re.compile('^{}.*'.format(folder_name))
    files = []
    for sub_dir in os.listdir(folder):
        if pattern.fullmatch(sub_dir):
            for file in os.listdir(os.path.join(folder, sub_dir)):
                if file.endswith(".csv"):
                    files.append(os.path.join(folder, sub_dir, file))

    return files


def get_results_from_files(superoperator_files):
    indices = ['protocol', 'swap', 'pg', 'pm', 'pn']
    result_df = pd.DataFrame(columns=indices)
    result_df = result_df.set_index(indices)
    for superoperator_file in superoperator_files:
        result_df = result_df.sort_index()
        df = pd.read_csv(superoperator_file, sep=';', index_col=[0, 1])
        index = tuple(df.iloc[0, df.columns.get_loc(index)] for index in indices)

        variables = ['written_to', 'avg_lde', 'avg_duration']
        for variable in variables:
            result_df.loc[index, variable] = df.iloc[0, df.columns.get_loc(variable)]

        result_df.loc[index, 'IIII'] = df['p'].loc[('IIII', False)]
        result_df.loc[index, 'IIZZ'] = df['p'].loc[('IIZZ', False)]

    print(result_df)


if __name__ == '__main__':
    name_csv = "./results/data_swap.csv"
    folder = "./results/sim_data"
    folder_names = ["superoperator_prb_dec"]
    files = []

    for folder_name in folder_names:
        files.extend(get_all_files_from_folder(folder, folder_name))

    get_results_from_files(files)
