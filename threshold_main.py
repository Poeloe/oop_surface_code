from matplotlib import pyplot as plt
from run_toric_2D_uf import multiprocess, multiple
from scipy import optimize
import numpy as np
import pandas as pd
import os

if __name__ == '__main__':

    save_result = 0

    folder = "../../../OneDrive - Delft University of Technology/MEP - thesis Mark/Simulations/"

    file_name = "uf_toric_pX_list_bucket_rand"
    plot_name = "uf_toric_pX_list_bucket_rand"

    lattices = []
    p = list(np.round(np.linspace(0.09, 0.11, 11), 6))
    Num = 50000
    plotn = 1000

    # Code #
    file_path = folder + "data/" + file_name + ".csv"
    if os.path.exists(file_path):
        data = pd.read_csv(file_path, header=0)
        data = data.set_index(["L", "p"])
    else:
        index = pd.MultiIndex.from_product([lattices, p], names=["L", "p"])
        data = pd.DataFrame(np.zeros((len(lattices)*len(p), 2)), index=index, columns=["N", "succes"])

    indices = data.index.values
    cols = ["N", "succes"]

    X = np.linspace(p[0], p[-1], plotn)
    f0 = plt.figure()

    # Simulate and save results to file
    for i, lati in enumerate(lattices):
        for pi in p:

            print("Calculating for L = ", str(lati), "and p =", str(pi))
            N_succes = multiprocess(lati, Num, 0, pi, 0, processes=4)
            # N_succes = multiple(lati, Num, 0, pi, 0)

            if any([(lati, pi) == a for a in indices]):
                data.loc[(lati, pi), "N"] += Num
                data.loc[(lati, pi), "succes"] += N_succes
            else:
                data.loc[(lati, pi), cols] = pd.Series([Num, N_succes]).values
                data = data.sort_index()

            if save_result:
                data.to_csv(file_path)

    print(data)

    # Fitting using scripy optimize curve_fot


    def fit_func(PL, pthres, A, B, C, D, nu, mu):
        p, L = PL
        x = (p - pthres) * L ** (1/nu)
        return A + B*x + C*x**2 + D * L**(-1/mu)

    g_T, T_m, T_M = 0.1, 0.09, 0.105
    g_A, A_m, A_M = 0, -np.inf, np.inf
    g_B, B_m, B_M = 0, -np.inf, np.inf
    g_C, C_m, C_M = 0, -np.inf, np.inf
    g_D, D_m, D_M = 0, -np.inf, np.inf
    gnu, num, nuM = 1.46, 1.2, 1.6
    gmu, mum, muM = 1, 0, 3

    par_guess = [g_T, g_A, g_B, g_C, g_D, gnu, gmu]
    bound = [(T_m, A_m, B_m, C_m, D_m, num, mum), (T_M, A_M, B_M, C_M, D_M, nuM, muM)]

    fitL = data.index.get_level_values('L')
    fitp = data.index.get_level_values('p')
    fitN = data.loc[:, "N"].values
    fitt = data.loc[:, "succes"].values

    par, pcov = optimize.curve_fit(fit_func, (fitp, fitL), fitt/fitN, par_guess, bounds=bound, sigma=fitN/max(fitN))
    perr = np.sqrt(np.diag(pcov))
    print("Least squared fitting on dataset results:")
    print("Threshold =", par[0], "+-", perr[0])
    print("A=", par[1], "B=", par[2], "C=", par[3])
    print("D=", par[4], "nu=", par[5], "mu=", par[6])

    # Plot all results from file (not just current simulation)
    all_lattices = []
    for lati, _ in indices:
        if lati not in all_lattices:
            all_lattices.append(lati)

    for i, lati in enumerate(all_lattices):
        plt.figure(f0.number)
        fp = data.loc[(lati)].index.values
        ft = [si / ni for ni, si in zip(data.loc[(lati), "N"].values, data.loc[(lati), "succes"].values)]
        w = data.loc[(lati), "N"].values/max(data.loc[(lati), "N"].values)
        plt.plot([q*100 for q in fp], ft, '.', color='C'+str(i), ms=10)
        plt.plot([x*100 for x in X], [fit_func((x, lati), *par) for x in X], '-', label=lati, color='C'+str(i), lw=2, alpha=0.8)

    plt.figure(f0.number)
    # plt.axvline(par[0]*100, ls="--", color="k", alpha=0.5, label="threshold = "+str(par[0]))
    plt.title("Threshold of " + plot_name)
    plt.xlabel("probability of Pauli X error (%)")
    plt.ylabel("decoding success rate (%)")
    plt.legend()
    plt.show()

    plt.figure()
    for L, p, N, t in zip(fitL, fitp, fitN, fitt):
        plt.plot((p-par[0])*L**(1/par[5]), t/N - par[4]*L**(-1/par[6]), '.')
    x = np.linspace(*plt.xlim(), plotn)
    plt.plot(x, par[1] + par[2]*x + par[3]*x**2, '--', color="C0", alpha=0.5)
    plt.xlabel("Rescaled error rate")
    plt.ylabel("Modified succces probability")
    plt.show()

    if save_result:
        data.to_csv(file_path)
        fname = folder + "./figures/" + file_name + ".pdf"
        f0.savefig(fname, transparent=True, format="pdf", bbox_inches="tight")
