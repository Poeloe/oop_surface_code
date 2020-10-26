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


def group_csv_files(filenames, error_values=None):
    indices_names = ["L", "p", "GHZ_success"]
    data = None
    for filename in filenames:
        if data is None:
            data = pd.read_csv(filename, header=0, float_precision='round_trip')
            data = data.set_index(indices_names)
            continue
        data_new = pd.read_csv(filename, header=0, float_precision='round_trip')
        data_new = data_new.set_index(indices_names)

        for index, columns in data_new.iterrows():
            for column, value in columns.items():
                if index in data.index:
                    data.at[index, column] = (value + data.at[index, column] if not pd.isna(data.at[index, column])
                                              else value)
                else:
                    data.at[index, column] = value

    if error_values is not None:
        idx = pd.IndexSlice
        data = data.loc[idx[:, error_values, :], :]

    data = data.sort_index()
    print(data)
    return data


def append_dataframes(dataframes):
    new_dataframe = pd.concat(dataframes)

    print(new_dataframe)


if __name__ == '__main__':
    name_csv = "./results/expedient_superoperators_swap.csv"
    folder = "./results/sim_data_new"
    folder_names = ["mwpm_expedient_swap_8",
                    "mwpm_expedient_swap_12",
                    "mwpm_expedient_swap_16",
                    "mwpm_expedient_swap_20"]
    error_values = [0.0016, 0.0017, 0.0018, 0.0019, 0.002]
    files = []

    for folder_name in folder_names:
        files.extend(get_all_files_from_folder(folder, folder_name))

    dataframe = group_csv_files(files, error_values)
    dataframe.to_csv(name_csv)
