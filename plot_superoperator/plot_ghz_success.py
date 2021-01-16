from matplotlib import pyplot as plt
import pandas as pd


def get_handles_and_labels(linestyles, ghz_values):
    handles = []
    labels = []

    handles.append(plt.Line2D((0, 1), (0, 0), color='g'))
    handles.append(plt.Line2D((0, 1), (0, 0), color='r'))
    handles.append(plt.Line2D((0, 1), (0, 0), color='b'))

    for linestyle in linestyles:
        handles.append(plt.Line2D((0, 1), (0, 0), linestyle=linestyle))

    labels.append("no post")
    labels.append("lower/upper")
    labels.append("all")

    for ghz_value in ghz_values:
        labels.append("GHZ success = " + str(ghz_value))

    return handles, labels


def plot_data(dfs, title):
    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)

    linestyles = ['solid', 'dashed', 'dashdot', 'dotted', 'solid']
    for df in dfs:
        for i, ghz_value in enumerate(ghz_values):
            if ghz_value in df.index.get_level_values("GHZ_success"):
                data = df.xs(ghz_value, level="GHZ_success")
                x_data = 100*data.index.get_level_values("p")
                y_data = data.success/data.N

                colour = 'r'
                if 'no_post' in df.name:
                    colour = 'g'
                if 'all' in df.name:
                    colour = 'b'

                ax.plot(x_data, y_data,
                        linestyle=linestyles[i],
                        color=colour,
                        marker='.',
                        label=('GHZ_success = ' + str(ghz_value)))

    handles, labels = get_handles_and_labels(linestyles, ghz_values)
    plt.ylabel("Decoding success rate")
    plt.xlabel("Probability of Pauli X error (%)")
    plt.title(title)
    plt.legend(handles, labels)


if __name__ == "__main__":

    files = ["6_no_post_uf.csv",
             "6_post_uf_unique.csv",
             "6_post_uf_all_unique.csv",
             "10_no_post_uf.csv",
             "10_post_uf_all_unique.csv",
             "10_post_uf_unique.csv"]

    df_6 = []
    df_10 = []
    for file in files:
        df = pd.read_csv(file, header=0, float_precision='round_trip')
        df = df.set_index(["L", "p", "GHZ_success"])
        df.name = file
        if '6' in file:
            df_6.append(df)
        else:
            df_10.append(df)

    ghz_values = [0.1, 0.5, 0.7, 0.9, 1.0]

    plot_data(df_6, "GHZ success rate comparison (UF, 10000 iterations, L=6)")
    plot_data(df_10, "GHZ success rate comparison (UF, 10000 iterations, L=10)")

    plt.show()