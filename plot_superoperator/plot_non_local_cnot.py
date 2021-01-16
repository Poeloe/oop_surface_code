import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats
from plot_superoperator.analyse_simulation_data import get_all_files_from_folder, confidence_interval
from matplotlib import colors as mcolors
import pickle
from scipy.stats import sem


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
        interval = stats.norm.interval(confidence, loc=mean, scale=np.std(fids_np))
        errors.append(interval) if not plus_mean else errors.append(interval[1])

    return errors


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


def combine_files(files, pkl_files):
    index_cols = [index[0] for index in pickle.load(open(pkl_files[0], "rb"))['index']]
    dataframes = [(file, pd.read_csv(file, sep=";", index_col=index_cols, float_precision='round_trip')) for file in
                  files]

    for filename in pkl_files:
        str_filename = filename.replace('.pkl', '').rstrip('0123456789')
        dataframe = [dataframe for file, dataframe in dataframes if file.replace(".csv", "") == str_filename][0]
        data = pickle.load(open(filename, 'rb'))
        index = tuple(index[1] for index in data["index"])
        dataframe.loc[index, 'fid_sem'] = sem(data['fidelities'])
        dataframe.loc[index, 'fid_std_l'] = confidence_interval(data['fidelities'], minus_mean=True)[0]
        dataframe.loc[index, 'fid_std_r'] = confidence_interval(data['fidelities'], minus_mean=True)[1]

    return dataframes


def plot_non_local_cnot_fidelity(dataframes, save_file_path, lde_values=None, spread=False):
    fig, ax = plot_style(title="Non-local CNOT gate", xlabel="Gate error probability", ylabel="Average fidelity")
    colors = [color for key, color in mcolors.TABLEAU_COLORS.items() if key not in ['tab:orange', 'tab:red']]

    color_count = 0
    for file, df in dataframes:
        sample = "Purified" if "na" not in file else "Natural abundance"
        df = df.reset_index()
        df = df.set_index('fixed_lde_attempts')
        for lde in set(df.index):
            lde_string = ("LDE attempts: " + str(lde) if all(df.loc[lde, 'pulse_duration'] > 0) else 'No decoupling')
            if lde_values is not None and lde not in lde_values and lde_string != "No decoupling":
                continue
            color_count = color_count + 1 if color_count < (len(colors) - 1) else 0
            color = colors[color_count]
            ax.errorbar(df.loc[lde, 'pg'],
                        df.loc[lde, 'avg_fidelity'],
                        yerr=None if not spread else [df.loc[lde, 'fid_std_l'],
                                                                        df.loc[lde, 'fid_std_r']],
                        ms=8,
                        fmt='-o',
                        capsize=8,
                        label="{} - {}".format(sample, lde_string), color=color)
            ax.set_xlim(0.051, 0)

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels, prop={'size': 18})
    plt.show()
    fig.savefig(save_file_path, transparent=False, format="pdf", bbox_inches="tight")


if __name__ == '__main__':
    save_file_path = '../results/thesis_files/draft_figures/non_local_gate.pdf'
    files, pkl_files = get_all_files_from_folder('../results/sim_data_4/non_local_gate',
                                                 ['purified', 'natural_abundance'],
                                                 True)

    dataframes = combine_files(files, pkl_files)
    plot_non_local_cnot_fidelity(dataframes, save_file_path, spread=False)
