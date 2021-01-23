import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import markers as mkrs
from itertools import product
from pprint import pprint


def get_unique_index_items(indices):
    index_unique_values = {}
    for key, value in indices.items():
        unique_values = sorted(set(value))
        if len(unique_values) > 1:
            index_unique_values[key] = unique_values

    return index_unique_values


def get_marker_index(marker_cols, run_dict):
    marker_ind = tuple()
    for value in marker_cols:
        marker_ind += (run_dict[value],)

    return marker_ind


def get_label_name(run_dict):
    value_translation = {"decoherence": "dec", "fixed_lde_attempts": "decoupling"}
    keep_key = ['pg', 'pn', 'pm', 'pm_1']
    name = ""
    for key, value in run_dict.items():
        if value:
            if key in value_translation:
                value = value_translation[key]
            name += "{}{}, ".format(key + "=" if key in keep_key else "", str(value).replace("_swap", ""))

    name = name.strip(", ")

    return name


def keep_rows_to_evaluate(df):
    df = df[df['cut_off_time'] != np.inf]
    for key, value in evaluate_values.items():
        df = df[df[key].isin(value)]

    return df


def identify_indices(df: pd.DataFrame):
    no_index_idicators = ['99', 'ghz', 'avg', 'sem', 'spread', 'IIII', 'written', 'cut', 'pulse']
    index_columns = {}
    for column in df:
        if all([indicator not in column for indicator in no_index_idicators]):
            unique_values = sorted(set(df[column]))
            if len(unique_values) > 1 or column in ['decoherence']:
                index_columns[column] = unique_values

    return index_columns


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


def scatter_plot(y_value, title, xlabel, ylabel, spread=False):
    colors = {}
    [colors.update({name: color}) for name, color in zip(index_dict['protocol_name'], mcolors.TABLEAU_COLORS)]
    points = list(mkrs.MarkerStyle.filled_markers)
    fig, ax = plot_style(title, xlabel, ylabel)
    i = 0
    protocol_markers = {}
    for index_tuple in product(*index_dict.values()):
        iteration_dict = dict(zip(index_dict.keys(), index_tuple))
        index = tuple(iteration_dict.values())

        if index in dataframe.index:
            protocol = iteration_dict['protocol_name']
            node = iteration_dict['node']
            dec = iteration_dict['decoherence']
            marker_index = get_marker_index(marker_index_cols, iteration_dict)
            if marker_index not in protocol_markers:
                protocol_markers[marker_index] = i
                i += 1
            color = colors[protocol]
            dataframe_new = dataframe.loc[index, :]
            style = 'none' if node == 'Purified' else 'full'
            error = {'ghz_fidelity': 'ghz', "IIII": "stab"}
            y_err = [[dataframe_new[error[y_value] + '_lspread']], [dataframe_new[error[y_value] + '_rspread']]]
            x_err = [[dataframe_new['dur_lspread']], [dataframe_new['dur_rspread']]]
            ax.errorbar(dataframe_new['avg_duration'],
                        dataframe_new[y_value],
                        yerr=None if not spread or not dec else y_err,
                        xerr=None if not spread or not dec else x_err,
                        marker=points[protocol_markers[marker_index]],
                        color=color,
                        ms=18 if dec else 8,
                        capsize=12,
                        label=get_label_name(iteration_dict),
                        fillstyle=style,
                        linestyle='')

    return fig, ax


if __name__ == '__main__':
    spread = True
    save = False
    file_name = '../results/circuit_data_NV_99_check.csv'
    dataframe = pd.read_csv(file_name, sep=';', float_precision='round_trip')
    pprint(identify_indices(dataframe))
    save_file_path_ghz = '../results/thesis_files/draft_figures/ghz_fidelity_vs_duration_check'
    save_file_path_stab = '../results/thesis_files/draft_figures/stab_fidelity_vs_duration_check'

    evaluate_values = {'decoherence':           [False, True],
                       'fixed_lde_attempts':    [0.0, 2000.0],
                       'node':                  ['Natural Abundance', 'Purified'],
                       'p_bell_success':        [0.0001],
                       'pg':                    [0.01],
                       'pm':                    [0.01],
                       'pm_1':                  ['0.001', '0.05', 'None'],
                       'protocol_name':         ['dyn_prot_4_14_1_swap',
                                                 'dyn_prot_4_4_1_swap',
                                                 'expedient_swap',
                                                 'plain_swap']
                       }
    dataframe = keep_rows_to_evaluate(dataframe)

    index_dict = identify_indices(dataframe)
    marker_index_cols = set(index_dict).difference(['node'])
    dataframe = dataframe.set_index(list(index_dict.keys()))

    fig, ax = scatter_plot("ghz_fidelity", "GHZ fidelity vs. duration", "Duration (s)",
                           "Fidelity", spread=spread)
    fig2, ax2 = scatter_plot("IIII", "Stabilizer fidelity vs. duration", "Duration (s)", "Fidelity", spread=spread)

    ax2.legend(prop={'size': 12})
    ax.legend(prop={'size': 12})
    plt.show()

    if spread:
        save_file_path_stab += "_spread"
        save_file_path_ghz += "_spread"

    if save:
        fig.savefig(save_file_path_ghz + ".pdf", transparent=False, format="pdf", bbox_inches="tight")
        fig2.savefig(save_file_path_stab + ".pdf", transparent=False, format="pdf", bbox_inches="tight")
