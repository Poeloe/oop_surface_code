'''
2020 Mark Shui Hu, QuTech

www.github.com/watermarkhu/oop_surface_code
_____________________________________________

'''
import argparse
from run_oopsc import add_kwargs
from oopsc.threshold.sim import sim_thresholds


def add_args(parser, args, group_name=None, description=None):

    if group_name:
        parser = parser.add_argument_group(group_name, description)
    for sid, lid, action, help, kwargs in args:
        parser.add_argument(sid, lid, action=action, help=help, **kwargs)


parser = argparse.ArgumentParser(
    prog="threshold_run",
    description="run a threshold computation",
    usage='%(prog)s [-h/--help] decoder lattice_type iters -l [..] -p [..] (lattice_size)'
)

arguments = [
    ["decoder", "store", str, "type of decoder - {mwpm/uf_uwg/uf/ufbb}", "d", dict()],
    ["lattice_type", "store", str, "type of lattice - {toric/planar}", "lt", dict()],
    ["iters", "store", int, "number of iterations - int", "i", dict()]
]

key_arguments = [
    ["-l", "--lattices", "store", "lattice sizes - verbose list int", dict(type=int, nargs='*', metavar="", required=True)],
    ["-p", "--perror", "store", "error rates - verbose list float", dict(type=float, nargs='*', metavar="", required=True)],
    ["-me", "--measurement_error", "store_true", "enable measurement error (2+1D) - toggle", dict()],
    ["-mt", "--multithreading", "store_true", "use multithreading - toggle", dict()],
    ["-nt", "--threads", "store", "number of threads", dict(type=int, metavar="")],
    ["-ma", "--modified_ansatz", "store_true", "use modified ansatz - toggle", dict()],
    ["-s", "--save_result", "store_true", "save results - toggle", dict()],
    ["-fn", "--file_name", "store", "plot filename", dict(default="thres", metavar="")],
    ["-f", "--folder", "store", "base folder path - toggle", dict(default=".", metavar="")],
    ["-pb", "--progressbar", "store_true", "enable progressbar - toggle", dict()],
    ["-fb", "--fbloom", "store", "pdc minimization parameter fbloom - float {0,1}", dict(type=float, default=0.5, metavar="")],
    ["-dgc", "--dg_connections", "store_true", "use dg_connections pre-union processing - toggle", dict()],
    ["-dg", "--directed_graph", "store_true", "use directed graph for evengrow - toggle", dict()],
    ["-db", "--debug", "store_true", "enable debugging heuristics - toggle", dict()],
]

add_args(parser, arguments)
add_kwargs(parser, key_arguments)
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
