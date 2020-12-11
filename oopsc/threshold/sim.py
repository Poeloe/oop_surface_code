'''
2020 Mark Shui Hu, QuTech

www.github.com/watermarkhu/oop_surface_code
_____________________________________________

'''
from .. import oopsc
from oopsc.superoperator import superoperator as so
from pprint import pprint
import multiprocessing as mp
import numpy as np
import pandas as pd
import sys, os


def read_data(file_path):
    try:
        data = pd.read_csv(file_path, header=0, float_precision='round_trip')
        indices = ["L", "p"] if "GHZ_success" not in data else ["L", "p", "GHZ_success"]
        return data.set_index(indices)
    except FileNotFoundError:
        print("File not found")
        exit()


def get_data(data, latts, probs, P_store=1):

    if not latts: latts = []
    if not probs: probs = []
    fitL = data.index.get_level_values("L")
    fitp = data.index.get_level_values("p")
    fitN = data.loc[:, "N"].values
    fitt = data.loc[:, "success"].values

    fitdata = [[] for i in range(4)]
    for L, P, N, t in zip(fitL, fitp, fitN, fitt):
        p = round(float(P)/P_store, 6)
        if all([N != 0, not latts or L in latts, not probs or p in probs]):
            fitdata[0].append(L)
            fitdata[1].append(p)
            fitdata[2].append(N)
            fitdata[3].append(t)

    return fitdata[0], fitdata[1], fitdata[2], fitdata[3]


def sim_thresholds(
        decoder,
        lattice_type="toric",
        lattices = [],
        perror = [],
        superoperator_filenames=[],
        superoperator_filenames_additional=None,
        GHZ_successes=[1.1],
        networked_architecture=False,
        iters = 0,
        measurement_error=False,
        multithreading=False,
        threads=None,
        save_result=True,
        file_name="thres",
        folder = ".",
        P_store=1000,
        debug=False,
        cycles=None,
        **kwargs
        ):
    '''
    ############################################
    '''
    run_oopsc = oopsc.multiprocess if multithreading else oopsc.multiple

    if measurement_error:
        from ..graph import graph_3D as go
    else:
        from ..graph import graph_2D as go

    sys.setrecursionlimit(100000)

    get_name = lambda s: s[s.rfind(".")+1:]
    g_type = get_name(go.__name__)
    d_type = get_name(decoder.__name__)
    full_name = f"{lattice_type}_{g_type}_{d_type}_{file_name}"

    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = folder + "/" + full_name + ".csv"

    progressbar = kwargs.pop("progressbar")

    data = None
    config = oopsc.default_config(**kwargs)

    superoperators = []
    if superoperator_filenames:
        perror = []
        for i, superoperator_filename in enumerate(superoperator_filenames):
            for GHZ_success in GHZ_successes:
                additional = [superoperator_filenames_additional[i]] if superoperator_filenames_additional is not None \
                    else None
                superoperator = so.Superoperator(superoperator_filename, GHZ_success,
                                                 additional_superoperators=additional)
                superoperators.append(superoperator)
                perror.append(superoperator.pg)

    # Simulate and save results to file
    for lati in lattices:

        if multithreading:
            if threads is None:
                threads = mp.cpu_count()
            graph = [oopsc.lattice_type(lattice_type, config, decoder, go, lati, cycles=cycles) for _ in range(threads)]
        else:
            graph = oopsc.lattice_type(lattice_type, config, decoder, go, lati, cycles=cycles)

        for i, pi in enumerate(perror):

            superoperator = None
            if superoperators:
                superoperator = superoperators[i]
                superoperator.reset_stabilizer_rounds()
                networked_architecture = bool(superoperator.pn) if not networked_architecture else True

            print("Calculating for L = {}{} and p = {}".format(lati, ', GHZ_success = ' +
                                                               str(superoperator.GHZ_success) if
                                                               superoperator else "", pi))

            oopsc_args = dict(
                paulix=pi,
                superoperator=superoperator,
                networked_architecture=networked_architecture,
                lattice_type=lattice_type,
                debug=debug,
                processes=threads,
                progressbar=progressbar
            )
            if measurement_error and not superoperator:
                oopsc_args.update(measurex=pi)
            output = run_oopsc(lati, config, iters, graph=graph, **oopsc_args)

            pprint(dict(output))
            print("")

            indices = [lattices, perror] if not superoperator else [lattices, perror, GHZ_successes]
            indices_names = ["L", "p"] if not superoperator else ["L", "p", "GHZ_success"]

            if data is None:
                if os.path.exists(file_path):
                    data = pd.read_csv(file_path, header=0, float_precision='round_trip')
                    data = data.set_index(indices_names)
                else:
                    columns = list(output.keys())
                    index = pd.MultiIndex.from_product(indices, names=indices_names)
                    data = pd.DataFrame(
                        np.zeros((len(lattices) * len(perror) * len(GHZ_successes), len(columns))), index=index,
                        columns=columns
                    )

            current_index = (lati, pi) if not superoperator else (lati, pi, superoperator.GHZ_success)

            if current_index in data.index:
                for key, value in output.items():
                    data.loc[current_index, key] += value
            else:
                for key, value in output.items():
                    data.loc[current_index, key] = value

            data = data.sort_index()
            if save_result:
                data.to_csv(file_path)

    print(data.to_string())

    if save_result:
        print("file saved to {}".format(file_path))
        data.to_csv(file_path)
