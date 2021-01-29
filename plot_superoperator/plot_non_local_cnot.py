import pickle
import numpy as np
import pandas as pd
from itertools import product
from scipy.stats import sem, norm
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
            index = tuple(v for _, v in key_value_index)
            avg_fidelities = [(4*fid + 1)/(4 + 1) for fid in sim_data['fidelities']]
            dataframe.loc[index, 'fid_sem'] = sem(sim_data['fidelities'])
            dataframe.loc[index, 'fid_entanglement'] = sum(sim_data['fidelities']) / len(sim_data['fidelities'])
            dataframe.loc[index, 'fid_std_l'] = confidence_interval(avg_fidelities, minus_mean=True)[0]
            dataframe.loc[index, 'fid_std_r'] = confidence_interval(avg_fidelities, minus_mean=True)[1]
            dataframe.loc[index, 'node'] = "Natural Abundance" if 'nat' in filename else "Purified"
        if full_dataframe is None:
            full_dataframe = dataframe
        else:
            full_dataframe = pd.concat([full_dataframe, dataframe])

    full_dataframe = full_dataframe.sort_index()

    if save_full:
        full_dataframe.to_csv(full_filename, sep=";")

    return full_dataframe


def filter_evaluate_values(values, x_axis):
    new_dict = {}
    for key, value in values.items():
        if len(value) > 1 and key != x_axis:
            new_dict[key] = value
        elif len(value) == 1 and key != x_axis:
            print("[+] {}={} holds for all displayed values".format(key, value.pop()))

    return new_dict


def plot_non_local_cnot_fidelity(df, x_axis, evaluate_values, save_file_path, spread=False,
                                 ent_fid=False):
    fig, ax = plot_style(title="Non-local CNOT gate", xlabel=x_axis, ylabel="Average fidelity")

    df = df.reset_index()
    df = keep_rows_to_evaluate(df, evaluate_values)
    evaluate_values = filter_evaluate_values(evaluate_values, x_axis)
    df = df.set_index(list(evaluate_values.keys()))
    df = df.sort_index()

    for index_tuple in product(*evaluate_values.values()):
        if index_tuple in df.index:
            x_axis_data = df.loc[index_tuple, x_axis]
            index_dict = dict(zip(evaluate_values.keys(), index_tuple))
            ax.errorbar(x_axis_data,
                        df.loc[index_tuple, 'avg_fidelity'],
                        yerr=None if not spread else [df.loc[index_tuple, 'fid_std_l'], df.loc[index_tuple, 'fid_std_r']],
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
    dataframes = combine_files(files, pkl_files, save_full)

    if spread:
        save_file_path += '_spread'
    plot_non_local_cnot_fidelity(dataframes, x_axis, evaluate_values, save_file_path + ".pdf", spread=spread,
                                 ent_fid=ent_fid)


if __name__ == '__main__':
    spread = False
    ent_fid = False
    save_full = True
    full_filename = '../notebooks/non_local_cnot_full.csv'
    save_file_path = '../results/thesis_files/draft_figures/non_local_gate_additional'
    files, pkl_files = get_all_files_from_folder('../results/sim_data_5/non_local_gate',
                                                 ['purified', 'natural_abundance'],
                                                 True)

    evaluate_values = {'node':                  [],
                       'decoherence':           [True],
                       'fixed_lde_attempts':    [2000],
                       'lde_success':           [],
                       'pg':                    [],
                       'pm':                    [0.01],
                       'pm_1':                  [0.01],
                       'pn':                    []
                       }
    x_axis = "pg"

    main(files, pkl_files, x_axis, evaluate_values, save_file_path, spread, ent_fid)
