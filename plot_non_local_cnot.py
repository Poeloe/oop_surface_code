import pandas as pd
from matplotlib import pyplot as plt


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


if __name__ == '__main__':
    csv_filename = './results/sim_data_3/non_local_gate/no_pulse.csv'
    csv_filename2 = './results/sim_data_3/non_local_gate/no_pulse_natural_abundance.csv'
    save_file_path = './results/thesis_files/draft_figures/non_local_gate.pdf'

    dataframe = pd.read_csv(csv_filename)
    dataframe2 = pd.read_csv(csv_filename2, sep=";")

    fig, ax = plot_style(title="Non-local CNOT gate", xlabel="Gate error probability (-)",
                         ylabel="Fidelity (-)")

    ax.plot(dataframe['pg'], dataframe['avg_fidelity'], '-x', label="Purified sample", ms=12)
    ax.plot(dataframe2['pg'], dataframe2['avg_fidelity'], '-o', label="Natural abundance sample", ms=12)
    ax.legend(prop={'size': 18})
    ax.set_xlim(0.051, 0)
    plt.show()
    fig.savefig(save_file_path, transparent=False, format="pdf", bbox_inches="tight")
