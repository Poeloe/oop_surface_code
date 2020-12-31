import pandas as pd
import os
import re
import pickle
import math

def confidence_interval(data, confidence=0.682):
    n = len(data)
    data_sorted = sorted(data)

    lower_bound = math.floor(n * ((1 - confidence) / 2))
    upper_bound = math.floor(n * (1 - (1 - confidence) / 2))

    return data_sorted[lower_bound], data_sorted[upper_bound]


def get_all_files_from_folder(folder, folder_name, pkl=False):
    pattern = re.compile('^{}.*'.format(folder_name))
    files = []
    pkl_files = []
    for sub_dir in os.listdir(folder):
        if pattern.fullmatch(sub_dir):
            for file in os.listdir(os.path.join(folder, sub_dir)):
                if file.endswith(".csv"):
                    files.append(os.path.join(folder, sub_dir, file))
                elif file.endswith(".pkl") and pkl:
                    pkl_files.append(os.path.join(folder, sub_dir, file))

    if pkl:
        return files, pkl_files

    return files


def get_results_from_files(superoperator_files, pkl_files, name_csv):
    indices = ['protocol_name', 'pg', 'pm', 'pm_1', 'pn', 'decoherence', 'p_bell_success', 'pulse_duration',
               'network_noise_type', 'no_single_qubit_error', 'basis_transformation_noise', 'cut_off_time',
               'probabilistic', 'fixed_lde_attempts']
    result_df = pd.DataFrame(columns=indices)
    result_df = result_df.set_index(indices)

    for superoperator_file in superoperator_files:
        pkl_file = (superoperator_file.replace(".csv", ".pkl") if superoperator_file.replace(".csv", ".pkl") in
                    pkl_files else None)
        full_data = pickle.load(open(pkl_file, "rb"))
        result_df = result_df.sort_index()
        df = pd.read_csv(superoperator_file, sep=';', index_col=[0, 1], float_precision='round_trip')
        if df.iloc[0, df.columns.get_loc('pulse_duration')] == 0:
            df.iloc[0, df.columns.get_loc('fixed_lde_attempts')] = 0
        df.iloc[0, df.columns.get_loc('protocol_name')] += "_na" if 'na' in superoperator_file else ""
        index = tuple(df.iloc[0, df.columns.get_loc(index)] for index in indices)

        variables = ['written_to', 'total_lde_attempts', 'avg_lde_attempts', 'total_duration', 'avg_duration',
                     'ghz_fidelity']
        for variable in variables:
            if variable in df:
                result_df.loc[index, variable] = df.iloc[0, df.columns.get_loc(variable)]

        interval_data = ['ghz_int', "dur_int", "stab_int"]
        for interval in interval_data:
            kind = interval.split(sep="_")[0]
            key = kind if "dur" in kind else kind + "_fid"
            result_df.loc[index, interval] = str(confidence_interval(full_data[key]))

        result_df.loc[index, '99_duration'] = confidence_interval(full_data["dur"], 0.98)[1]

        result_df.loc[index, 'IIII'] = df['p'].loc[('IIII', False)]
        result_df.loc[index, 'IIZZ'] = df['p'].loc[('IIZZ', False)]

    result_df = result_df.sort_index()
    result_df.to_csv(name_csv, sep=';')
    print(result_df)


if __name__ == '__main__':
    name_csv = "./results/circuit_data_NV_info.csv"
    folder = "./results/sim_data_4"
    folder_name = "superoperator_cutoff_info"

    files, pkl_files = get_all_files_from_folder(folder, folder_name, pkl=True)

    get_results_from_files(files, pkl_files, name_csv)
