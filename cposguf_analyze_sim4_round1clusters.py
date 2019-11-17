import cposguf_cluster_actions as cca
from cposguf_run import sql_connection, fetch_query
import graph_objects as go
import toric_error as te
from collections import defaultdict as dd
from cluster_per_size import coordinates
import numpy as np
import pickle
import os


limit = None
plotnum = 25
maxfetch = 5000
L = [8] # + 4 * i for i in range(10)]
P = [(90 + i)/1000 for i in range(21)]
file = "sim4_r1c_data_gauss12_44"


##############################################
# definitions:


def d0(): return [0,0]
def d1(): return [[0,0], [0,0]]
def d2(): return dd(d1)

def load_obj(name):
    with open(f"/data/{name}.pkl", "rb") as f:
        return pickle.load(f)

def save_obj(obj, name):    # Save object to a pickled file
    with open(f"/data/{name}.pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def get_count(clusters, clist, size):
    """
    returns a defaultdict of lists of clusters countaining the tuples of the qubits
    """
    count = [0 for i in clist]
    clusters = cca.listit(clusters)

    # Return to principal cluster
    for i, cluster in enumerate(clusters):
        clusters[i] = cca.principal(cluster, ydim=size, xdim=size)
    for cluster in clusters:
        augs, cmid = [], []
        for rot_i in range(4):
            dim = cca.max_dim(cluster)
            augs.append(cluster)
            cmid.append(dim[1])
            fcluster = frozenset(cluster)
            if fcluster in clist:
                count[clist.index(fcluster)] += 1
                break
            mcluster = cca.mirror(cluster, dim[0])
            mdim = cca.max_dim(mcluster)
            augs.append(mcluster)
            cmid.append(mdim[1])
            fmcluster = frozenset(mcluster)
            if fmcluster in clist:
                count[clist.index(fmcluster)] += 1
                break
            cluster = cca.rotate(cluster, dim[0])
        else:
            ftupcluster = frozenset(augs[cmid.index(min(cmid))])
            count[clist.index(ftupcluster)] += 1

    return count


###########################################################

if os.path.exists(file + ".pkl"):           # Load data_base if pickled file exists
    print("loading data " + file)
    data = load_obj(file)
    data_p, data_n = data["data_p"], data["data_n"]
    countp, countn = data["countp"], data["countn"]
else:                                       # Initate database
    data_p, data_n = [dd(d2) for _ in range(2)]
    countp, countn = [dd(d0) for _ in range(2)]

minl, maxl = 4, 4
clist = coordinates(minl, maxl)

for lattice in L:
    for p in P:
        temp_p, temp_n = [[], []], dd(list)

        con, cur = sql_connection()
        print("\nGetting count of L{}, p{}...".format(lattice, p))
        cur.execute("SELECT tree_wins, list_wins FROM cases WHERE lattice = {} and p = {}".format(lattice, p))
        tlcount = cur.fetchone()
        countp[(lattice, p)] = [tlcount[0], tlcount[1]]

        cur.execute(fetch_query("COUNT(*)", lattice, p))
        num = cur.fetchone()[0]
        print("fetching {} simulations...".format(num))

        cur.execute(fetch_query("ftree_tlist, seed", lattice, p))
        sims = [cur.fetchone()]
        graph = go.init_toric_graph(lattice)

        fetched = 1
        while sims != [None]:

            print("{:0.1f}%".format(fetched/num*100))
            sims += cur.fetchmany(maxfetch)
            fetched += maxfetch

            for type, seed in sims:

                # Get errors from seed
                te.apply_random_seed(seed)
                te.init_pauli(graph, pX=float(p))
                errors = [
                    graph.E[(0, y, x, td)].qID[1:]
                    for y in range(lattice) for x in range(lattice) for td in range(2)
                    if graph.E[(0, y, x, td)].state
                ]
                n = len(errors)

                # Get clusters from array data
                clusters = cca.get_clusters_from_list(errors, lattice)

                # Remove outlier clusters
                clusters = [
                    cluster
                    for cluster in clusters.values()
                    if len(cluster) >= minl and len(cluster) <= maxl
                ]

                # Store cluster count to temp dictionaries
                count = get_count(clusters, clist, lattice)
                temp_p[type].append(count)
                temp_n[(n, type)].append(count)

                graph.reset()
            sims = [cur.fetchone()]
        else:
            print("100%")
            cur.close()
            con.close()

        # Get mean and std of p data distribution and store to database
        cl0 = list(map(list, zip(*temp_p[0])))
        cl1 = list(map(list, zip(*temp_p[1])))

        for list0, list1, key in zip(cl0, cl1, clist):
            data_p[key][(lattice, p)][0] = [np.mean(list0), np.var(list0)]
            data_p[key][(lattice, p)][1] = [np.mean(list1), np.var(list1)]

        # Get mean and std of n data distribution and combine with existing mean and std of previous analysis

        # (n, type), nlists = sorted(list(temp_n.items()), key=lambda kv: len(kv[1]), reverse=True)[0]
        # oc = countp[(lattice, p)][type]
        # nc = len(nlists)
        # cl = list(map(list, zip(*nlists)))
        # fig = plt.figure()
        # plt.hist(cl[0], bins=range(min(cl[0]), max(cl[0]) + 1, 1), normed=True)
        # x_axis = np.arange(min(cl[0]), max(cl[0]), 0.01)
        # plt.plot(x_axis, stats.norm.pdf(x_axis,np.mean(cl[0]),np.std(cl[0])))
        # plt.xlabel("Population on lattice")
        # plt.ylabel("Normalized occurance")
        # name = f"Occurance of R1-cluster 0 in L{lattice}, n{n}"
        # plt.title(name)
        # plt.show()
        # fig.savefig(name + ".pdf", transparent=True, format="pdf", bbox_inches="tight")
        # exit()

        for (n, type), nlists in temp_n.items():
            oc = countp[(lattice, p)][type]
            nc = len(nlists)
            cl = list(map(list, zip(*nlists)))

            for list0, key in zip(cl, clist):
                oldmu, oldva = data_n[key][(lattice, n)][type]
                newmu = (oc*oldmu + nc*np.mean(list0))/(oc+nc)
                newva = (oc*(oldmu**2 + oldva) + nc*(newmu**2 + np.var(list0)))/(oc+nc) - newmu**2
                data_n[key][(lattice, n)][type] = [newmu, newva]
            countn[(lattice, n)][type] += nc

        print("Saving data...")
        data = {    # Save to single data file
            "data_p": data_p,
            "data_n": data_n,
            "countp": countp,
            "countn": countn
        }
        save_obj(data, file)
