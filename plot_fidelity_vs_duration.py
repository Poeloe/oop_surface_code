import pandas as pd
from matplotlib import pyplot as plt
from itertools import product


if __name__ == '__main__':
    file_name = './results/circuit_data_NV.csv'

    dataframe = pd.read_csv(file_name, sep=';', index_col=['protocol_name', 'fixed_lde_attempts', 'pulse_duration'])
    points = ["o", "s", "v", "D", "p", "^", "h", "X", "<", "P", "*", ">", "H", "d"]
    protocol_names = sorted(set([name[0] for name in dataframe.index]))
    lde_attempts = sorted(set([index[1] for index in dataframe.index]))
    pulse_durations = sorted(set([index[2] for index in dataframe.index]))
    colors = {}
    [colors.update({name: color}) for name, color in zip(protocol_names, ['b', 'g', 'r', 'y'])]

    prev_protocol = None
    i = 0
    for protocol, lde, pulse_duration in product(protocol_names, lde_attempts, pulse_durations):
        idx = pd.IndexSlice
        index = idx[protocol, lde, pulse_duration]
        if index in dataframe.index:
            color = colors[protocol]
            dataframe_new = dataframe.loc[index, :]
            i = i + 1 if protocol == prev_protocol else 0
            plt.plot(dataframe_new['avg_duration'],
                     dataframe_new['ghz_fidelity'],
                     points[i],
                     color=color,
                     ms=12,
                     label="{}-{}".format(protocol, str(lde)),
                     fillstyle='full')
            prev_protocol = protocol

    plt.title("Fidelity vs. Duration")
    plt.xlabel("Duration (s)")
    plt.ylabel("Fidelity (-)")
    plt.legend()
    plt.show()
