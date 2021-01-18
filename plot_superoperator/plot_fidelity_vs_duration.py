import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from itertools import product


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
    [colors.update({name: color}) for name, color in zip(protocol_names, mcolors.TABLEAU_COLORS)]
    points = ["o", "s", "v", "D", "p", "^", "h", "X", "<", "P", "*", ">", "H", "d"]
    fig, ax = plot_style(title, xlabel, ylabel)
    i = 0
    protocol_markers = {}
    for protocol, node, lde, pulse_duration, dec in product(protocol_names, nodes, lde_attempts, pulse_durations,
                                                            decoherence):

        idx = pd.IndexSlice
        index = idx[protocol, node, lde, pulse_duration, dec]
        if index in dataframe.index:
            marker_index = (protocol, lde, dec)
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
                        label="{}, {}{}{}".format(protocol.replace('_swap', '').replace('_na', ''),
                                                  node,
                                                  ', ' + str(int(lde)) if lde else "",
                                                  ', decoherence' if dec else ''),
                        fillstyle=style,
                        linestyle='')

    return fig, ax


if __name__ == '__main__':
    file_name = './results/circuit_data_NV_info.csv'
    spread = True
    save_file_path_ghz = './results/thesis_files/draft_figures/ghz_fidelity_vs_duration'
    save_file_path_stab = './results/thesis_files/draft_figures/stab_fidelity_vs_duration'
    lde_skip = [3000, 5000]
    protocol_skip = ['stringent_swap']

    dataframe = pd.read_csv(file_name, sep=';', index_col=['protocol_name', 'node', 'fixed_lde_attempts',
                                                           'pulse_duration', 'decoherence'])
    protocol_names = sorted(set([index[0] for index in dataframe.index]).difference(protocol_skip))
    nodes = sorted(set([index[1] for index in dataframe.index]))
    lde_attempts = sorted(set([index[2] for index in dataframe.index]).difference(lde_skip))
    pulse_durations = sorted(set([index[3] for index in dataframe.index]))
    decoherence = sorted(set([index[4] for index in dataframe.index]))

    fig, ax = scatter_plot("ghz_fidelity", "GHZ fidelity vs. duration", "Duration (s)",
                           "Fidelity", spread=spread)
    fig2, ax2 = scatter_plot("IIII", "Stabilizer fidelity vs. duration", "Duration (s)", "Fidelity", spread=spread)

    ax2.legend(prop={'size': 12})
    ax.legend(prop={'size': 12})
    plt.show()

    if spread:
        save_file_path_stab += "_spread"
        save_file_path_ghz += "_spread"

    fig.savefig(save_file_path_ghz + ".pdf", transparent=False, format="pdf", bbox_inches="tight")
    fig2.savefig(save_file_path_stab + ".pdf", transparent=False, format="pdf", bbox_inches="tight")
