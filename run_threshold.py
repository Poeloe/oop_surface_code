'''
2020 Mark Shui Hu, QuTech

www.github.com/watermarkhu/oop_surface_code
_____________________________________________

'''
import argparse
from run_oopsc import add_args, add_kwargs
from oopsc.threshold.sim import sim_thresholds


def add_arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    key_arguments = [
        ["-de", "--decoder", "store", "type of decoder - {mwpm/uf_uwg/uf/ufbb}",
         dict(type=str, metavar="", default='mwpm')],
        ["-lt", "--lattice_type", "store", "type of lattice - {toric/planar}",
         dict(type=str, metavar="", default='toric')],
        ["-ite", "--iters", "store", "number of iterations - int",
         dict(type=int, metavar="", default=10000)],
        ["-l", "--lattices", "store", "lattice sizes - verbose list int",
         dict(type=int, nargs='*', metavar="", default=[8])],
        ["-pe", "--perror", "store", "error rates - verbose list float",
         dict(type=float, nargs='*', metavar="", default=[])],
        ["-so", "--superoperator_filenames", "store",
         "Use superoperator as error input - list of superoperator filenames",
         dict(type=str, nargs='*', metavar="")],
        ["-failed_so", "--superoperator_filenames_failed", "store",
         "Use superoperator as error input - list of superoperator filenames",
         dict(type=str, nargs='*', metavar="", default=None)],
        ["-add_so", "--superoperator_filenames_additional", "store",
         "Use superoperator as error input - list of superoperator filenames",
         dict(type=str, nargs='*', metavar="", default=None)],
        ["-add_failed_so", "--superoperator_filenames_additional_failed", "store",
         "Use superoperator as error input - list of superoperator filenames",
         dict(type=str, nargs='*', metavar="", default=None)],
        ["-nw", "--networked_architecture", "store_true",
         "Force to run threshold simulations with a networked architecture", dict()],
        ["-me", "--measurement_error", "store_true", "enable measurement error (2+1D) - toggle", dict()],
        ["-mt", "--multithreading", "store_true", "use multithreading - toggle", dict()],
        ["-nt", "--threads", "store", "number of threads", dict(type=int, metavar="")],
        ["-ma", "--modified_ansatz", "store_true", "use modified ansatz - toggle", dict()],
        ["-sv", "--save_result", "store_true", "save results - toggle", dict()],
        ["-fname", "--file_name", "store", "plot filename", dict(default="thres", metavar="")],
        ["-f", "--folder", "store", "base folder path - toggle", dict(default=".", metavar="")],
        ["-pb", "--progressbar", "store_true", "enable progressbar - toggle", dict()],
        ["-fb", "--fbloom", "store", "pdc minimization parameter fbloom - float {0,1}",
         dict(type=float, default=0.5, metavar="")],
        ["-dgc", "--dg_connections", "store_true", "use dg_connections pre-union processing - toggle", dict()],
        ["-dg", "--directed_graph", "store_true", "use directed graph for evengrow - toggle", dict()],
        ["-db", "--debug", "store_true", "enable debugging heuristics - toggle", dict()],
        ["-GHZ", "--GHZ_successes", "store", "specify the percentage of GHZ states that are successfully created"
                                             " (works only with superoperator) - float [0-1]",
         dict(type=float, nargs='*', metavar="", default=[1.1])],
        ["-cy", "--cycles", "store", "Amount of stabilizer cycles (default=l)", dict(type=int, metavar="",
                                                                                     default=None)]
    ]

    add_kwargs(parser, key_arguments)

    return parser


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="threshold_run",
        description="run a threshold computation",
        usage='%(prog)s [-h/--help] decoder lattice_type iters -l [..] -p [..] (lattice_size)'
    )
    parser = add_arguments(parser)

    args=vars(parser.parse_args())
    decoder = args.pop("decoder")

    decoders = __import__("oopsc.decoder", fromlist=[decoder])
    decode = getattr(decoders, decoder)

    decoder_names = {
        "mwpm":     "minimum weight perfect matching (blossom5)",
        "uf":       "union-find",
        "uf_uwg":   "union-find non weighted growth",
        "ufbb":     "union-find balanced bloom"
    }
    decoder_name = decoder_names[decoder] if decoder in decoder_names else decoder
    print(f"{'_'*75}\n\ndecoder type: " + decoder_name)

    sim_thresholds(decode, **args)
