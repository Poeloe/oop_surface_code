import run_toric_2D_uf as rt2u
import run_toric_2D_mwpm as rt2m
import time

if __name__ == '__main__':

    t0 = time.time()
    size = 10
    pX = 0.11
    pZ = 0.0
    pE = 0.0
    iters = 20000

    plot_load = 1
    save_file = 0
    filename = None
    pauli_file = filename + "_pauli" if filename is not None else None
    # pauli_file = "test_pauli"
    erasure_file = filename + "_erasure" if filename is not None else None

    output = rt2u.single(size, pE, pX, pZ, save_file, erasure_file, pauli_file, plot_load)
    # output = rt2u.multiple(size, iters, pE, pX, pZ, plot_load=plot_load)
    # output = rt2u.multiprocess(size, iters, pE, pX, pZ, 4)


    print("time taken =", time.time()-t0)
    print("p = " + str(output/iters*100) + "%")
