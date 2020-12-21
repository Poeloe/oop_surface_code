import pandas as pd
from matplotlib import pyplot as plt


if __name__ == '__main__':
    csv_filename = './results/sim_data_3/non_local_gate/no_pulse.csv'

    dataframe = pd.read_csv(csv_filename)

    plt.plot(dataframe['pg'], dataframe['avg_fidelity'])
    plt.show()
