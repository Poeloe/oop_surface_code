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


def scatter_plot(y_value, title, xlabel, ylabel):
    colors = {}
    [colors.update({name: color}) for name, color in zip(protocol_names, mcolors.TABLEAU_COLORS)]
    points = ["o", "s", "v", "D", "p", "^", "h", "X", "<", "P", "*", ">", "H", "d"]
    fig, ax = plot_style(title, xlabel, ylabel)
    prev_protocol = None
    i = 0
    for protocol, lde, pulse_duration in product(protocol_names, lde_attempts, pulse_durations):
        idx = pd.IndexSlice
        index = idx[protocol, lde, pulse_duration]
        if index in dataframe.index:
            color = colors[protocol]
            dataframe_new = dataframe.loc[index, :]
            i = i + 1 if protocol == prev_protocol else 0
            style = 'none' if 'NA' not in protocol else 'full'
            ax.plot(dataframe_new['avg_duration'],
                    dataframe_new[y_value],
                    points[i],
                    color=color,
                    ms=18,
                    label="{}-{}".format(protocol, str(lde)),
                    fillstyle=style)
            prev_protocol = protocol

    return fig, ax


if __name__ == '__main__':
    file_name = './results/circuit_data_NV.csv'
    save_file_path_ghz = './results/thesis_files/draft_figures/ghz_fidelity_vs_duration.pdf'
    save_file_path_stab = './results/thesis_files/draft_figures/stab_fidelity_vs_duration.pdf'

    dataframe = pd.read_csv(file_name, sep=';', index_col=['protocol_name', 'fixed_lde_attempts', 'pulse_duration'])
    protocol_names = sorted(set([name[0] for name in dataframe.index]))
    lde_attempts = sorted(set([index[1] for index in dataframe.index]))
    pulse_durations = sorted(set([index[2] for index in dataframe.index]))

    fig, ax = scatter_plot("ghz_fidelity", "GHZ fidelity vs. Duration", "Duration (s)", "Fidelity (-)")
    fig2, ax2 = scatter_plot("IIII", "Stabilizer fidelity vs. Duration", "Duration (s)", "Fidelity (-)")

    ax2.legend(prop={'size': 12})
    ax.legend(prop={'size': 12})
    plt.show()

    fig.savefig(save_file_path_ghz, transparent=False, format="pdf", bbox_inches="tight")
    fig2.savefig(save_file_path_stab, transparent=False, format="pdf", bbox_inches="tight")