import os
import pickle
import numpy as np
import pandas as pd
from itertools import product
from scipy.stats import sem, norm
from collections import defaultdict
from matplotlib import pyplot as plt
from plot_superoperator.analyse_simulation_data import get_all_files_from_folder, confidence_interval
from plot_superoperator.plot_fidelity_vs_duration import get_label_name


def mean_confidence_interval(data, confidence=0.682, plus_mean=False):
    if len(set(data)) == 1:
        return "Not enough data"
    if any([type(el) != list for el in data]):
        data = [data]
    errors = []
    for fids in data:
        fids_np = np.array(fids)
        n = len(fids_np)
        mean = np.mean(fids_np)
        interval = norm.interval(confidence, loc=mean, scale=np.std(fids_np))
        errors.append(interval) if not plus_mean else errors.append(interval[1])

    return errors


def keep_rows_to_evaluate(df, ev_values):
    for key, values in ev_values.items():
        values = values + [str(i) for i in values]  # Usage: if dataframe parses the values as strings

        if values and 'REMOVE' not in values:
            df = df[df[key].isin(values)]
        elif 'REMOVE' not in values:
            ev_values[key] = set(df[key])

    if df.empty:
        raise ValueError("The combination of values does not exist in the dataframe!")
    return df


def plot_style(title=None, xlabel=None, ylabel=None, **kwargs):
    fig, ax = plt.subplots(figsize=(20, 14))
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.subplots_adjust(left=0.08, bottom=0.08, right=.95, top=.95)
    ax.grid(color='w', linestyle='-', linewidth=2)
    ax.set_title(title, fontsize=34)
    ax.set_xlabel(xlabel, fontsize=30)
    ax.set_ylabel(ylabel, fontsize=30)
    for key, arg in kwargs.items():
        func = getattr(ax, f"set_{key}")
        func(arg)
    ax.patch.set_facecolor('0.95')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    return fig, ax


def combine_files(files, pkl_files, save_full):
    index_cols = [index_tuple[0] for index_tuple in list(pickle.load(open(pkl_files[0], "rb")).keys())[0]]
    full_dataframe = None

    for filename in files:
        dataframe = pd.read_csv(filename, sep=";", index_col=index_cols, float_precision='round_trip')
        data = pickle.load(open(filename.replace('.csv', '.pkl'), 'rb'))
        for key_value_index, sim_data in data.items():
            index = tuple(v for _, v in sorted(key_value_index, key=lambda i: index_cols.index(i[0])))
            if index not in dataframe.index:
                continue
            avg_fidelities = [(4*fid + 1)/(4 + 1) for fid in sim_data['fidelities']]
            dataframe.loc[index, 'fid_sem'] = sem(sim_data['fidelities'])
            dataframe.loc[index, 'fid_entanglement'] = sum(sim_data['fidelities']) / len(sim_data['fidelities'])
            dataframe.loc[index, 'fid_std_l'] = confidence_interval(avg_fidelities, minus_mean=True)[0]
            dataframe.loc[index, 'fid_std_r'] = confidence_interval(avg_fidelities, minus_mean=True)[1]
            dataframe.loc[index, 'node'] = "Natural Abundance" if 'nat' in filename else "Purified"
        if full_dataframe is None:
            full_dataframe = dataframe
        else:
            full_dataframe = full_dataframe.append(dataframe, sort=True)

    full_dataframe = full_dataframe.sort_index()

    if save_full:
        full_dataframe.to_csv(full_filename, sep=";")

    return full_dataframe


def filter_evaluate_values(df, values, x_axis):
    new_dict = {}
    for key, value in values.items():
        if len(value) > 1 and key != x_axis:
            new_dict[key] = value

    df.set_index(list(new_dict.keys()), inplace=True)
    df.sort_index(inplace=True)
    indices = []
    index_dicts = []

    for index in product(*new_dict.values()):
        index = index if len(index) > 1 else index[0]
        if index in df.index:
            index_dict = dict(zip(new_dict.keys(), index if type(index) != str else [index]))
            indices.append(index)
            index_dicts.append(index_dict)

    # Show the fixed parameters
    print("The following values are fixed:")
    print_parameters = []
    for column in df:
        val = set(df[column])
        if len(val) == 1:
            print_parameters.append("\t[+] {}={}\n".format(column, list(val)[0]))

    # Remove keys from the dicts that have a fixed value (such that the legend only shows varying parameters)
    full_dict = defaultdict(list)
    [full_dict[k].append(v) for d in index_dicts for k, v in d.items()]
    for k, v in full_dict.items():
        if len(set(v)) == 1:
            [d.pop(k) for d in index_dicts]
            print_parameters.append("\t[+] {}={}\n".format(k, v[0]))
    print(*sorted(print_parameters))

    return df, indices, index_dicts


def pre_process_results(df, values, x_axis):
    df = df.reset_index()
    df = keep_rows_to_evaluate(df, values)
    df, indices, index_dicts = filter_evaluate_values(df, values, x_axis)

    return df, indices, index_dicts


def plot_non_local_cnot_fidelity(df, x_axis, evaluate_values, save_file_path, spread=False,
                                 ent_fid=False):
    fig, ax = plot_style(title="Non-local CNOT gate", xlabel=x_axis, ylabel="Average fidelity")
    df, indices, index_dicts = pre_process_results(df, evaluate_values, x_axis)

    for index_tuple, index_dict in zip(indices, index_dicts):
        x_axis_data = df.loc[index_tuple, x_axis]
        ax.errorbar(x_axis_data,
                    df.loc[index_tuple, 'avg_fidelity'],
                    yerr=None if not spread else [df.loc[index_tuple, 'fid_std_l'],
                                                  df.loc[index_tuple, 'fid_std_r']],
                    ms=8,
                    fmt='-o',
                    capsize=8,
                    label=get_label_name(index_dict))
        if ent_fid:
            ax.errorbar(x_axis_data,
                        df.loc[index_tuple, 'fid_entanglement'],
                        fmt='-o',
                        label="{} - {}".format(get_label_name(index_dict), "$F_{e}$"))
        ax.set_xlim(max(x_axis_data) + 0.001, min(x_axis_data) - 0.001)

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels, prop={'size': 18})
    plt.show()
    fig.savefig(save_file_path, transparent=False, format="pdf", bbox_inches="tight")


def main(files, pkl_files, x_axis, evaluate_values, save_file_path, spread, ent_fid):
    if not os.path.exists(full_filename) or save_full:
        full_dataframe = combine_files(files, pkl_files, save_full)
    else:
        index_cols = [index_tuple[0] for index_tuple in list(pickle.load(open(pkl_files[0], "rb")).keys())[0]]
        full_dataframe = pd.read_csv(full_filename, sep=';', index_col=index_cols, float_precision="round_trip")

    if spread:
        save_file_path += '_spread'
    plot_non_local_cnot_fidelity(full_dataframe, x_axis, evaluate_values, save_file_path + ".pdf", spread=spread,
                                 ent_fid=ent_fid)


if __name__ == '__main__':
    spread = True
    ent_fid = False
    save_full = False
    full_filename = '../notebooks/non_local_cnot_full.csv'
    save_file_path = '../results/thesis_files/draft_figures/non_local_gate_additional'
    files, pkl_files = get_all_files_from_folder('../results/sim_data_5/non_local_gate',
                                                 ['purified', 'natural_abundance'],
                                                 True)

    evaluate_values = {'node':                  [],
                       'decoherence':           [True],
                       'fixed_lde_attempts':    [2000],
                       'lde_success':           [0.0001],
                       'pg':                    [0.01],
                       'pm':                    [0.01],
                       'pm_1':                  [],
                       'pn':                    [0.1],
                       'T1_lde':                [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001],
                       }
    x_axis = "T1_lde"

    main(files, pkl_files, x_axis, evaluate_values, save_file_path, spread, ent_fid)
