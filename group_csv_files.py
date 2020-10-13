import pandas as pd
import os
import re


def get_all_files_from_folder(folder, folder_name):
    pattern = re.compile('^{}.*'.format(folder_name))
    files = []
    for sub_dir in os.listdir(folder):
        if pattern.fullmatch(sub_dir):
            for file in os.listdir(os.path.join(folder, sub_dir)):
                files.append(os.path.join(folder, sub_dir, file))

    return files


def group_csv_files(filenames):
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
                data.at[index, column] += value

    print(data)
    return data


def append_dataframes(dataframes):
    new_dataframe = pd.concat(dataframes)

    print(new_dataframe)


if __name__ == '__main__':
    name_csv = "./results/expedient_superoperators_naomi.csv"
    folder = "./results/sim_data"
    folder_names = ["mwpm_expedient_sim_8", "mwpm_expedient_sim_12", "mwpm_expedient_sim_16", "mwpm_expedient_sim_20"]
    dataframes = []

    for folder_name in folder_names:
        files = get_all_files_from_folder(folder, folder_name)
        dataframes.append(group_csv_files(files))

    total_dataframe = pd.concat(dataframes)
    total_dataframe.to_csv(name_csv)
