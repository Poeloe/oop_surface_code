# Quantum Circuit Simulator

This module is a density matrix based (noisy) quantum circuit simulator and is, among others, capable of calculating 
the resulting superoperator.

## Requirements

* Python 3.8
* Numpy 
* Scipy

## Basic Operations

### Initialise QuantumCircuit object
Using the module starts with creating an QuantumCircuit object. This object can be created with the following 
parameters:

* `num_qubits` `(int)`: Number of qubits that the system should (initially) contain.

* `init_type` `(int, optional, default=0)`: How the system should be initialised.
    * `0` : All qubits are initialised in the perfect `|0>` state.
    * `1` : First qubit in `|+>` state, rest in `|0>`. 
    * `2` : Qubits in fully entangled state (perfect Bell pair between each pair of qubits, `num_qubits` should be 
    even).
    * `3` : Same as `1` only a `CNOT`-gate is applied to all qubits with top qubit as control qubit.

* `noise` `(bool, optional, default=False)` : Specifies if the Quantum Circuit should be noisy in general. This means 
that, whenever not specified differently, this parameter will hold for each operation that is applied.

* `pg` `(float [0-1], optional, default=0.001)` : General parameter to specify the amount of gate noise that should be 
applied when a gate operation is done. This value will hold for each operation, unless differently specified.  

* `pm` `(float [0-1], optional, default=0.001)` : General parameter to specify the amount of measurement noise that 
should be applied when a measurement is performed. This value will hold for each measurement, unless differently 
specified.

* `pg` `(float [0-1], optional, default=0.1)` : General parameter to specify the amount of network noise that should 
be applied when a Bell pair is created. This value will hold for each Bell pair creation, unless differently specified.

And QuantumCircuit object can thus for example be initialised as:
    
        qc = QuantumCircuit(num_qubits=2, init_type=0, noise=True, pg=0.001, pm=0.001, pn=0.1)

### Perform operation on the QuantumCircuit Object

When a QuantumCircuit object is created, operation can be applied to it in order to simulate a real world quantum 
circuit. The most common operations are:

* `apply_1_qubit_gate`:                 
                
        Applies a one qubit gate to the specified target qubit. This will update the density
        matrix of the system accordingly.

                Parameters
                ----------
                gate : ndarray
                    Array of dimension 2x2, examples are the well-known pauli matrices (X, Y, Z)
                tqubit : int
                    Integer that indicates the target qubit. Note that the qubit counting starts at
                    0.
                noise : bool, optional, default=None
                    Determines if the gate is noisy. When the QuantumCircuit object is initialised
                    with the 'noise' parameter to True, this parameter will also evaluate to True if
                    not specified otherwise.
                pg : float [0-1], optional, default=None
                    Specifies the amount of gate noise if present. If the QuantumCircuit object is
                    initialised with a 'pg' parameter, this will be used if not specified otherwise
                draw : bool, optional, default=True
                    If true, the specified gate will appear when the circuit is visualised.
                user_operation : bool, optional, default=True
                True if the user has requested the method and (else) False if it was invoked by an internal
                method.

    For which the known single-qubit gates are already provided:
    * `X`
    * `Z`
    * `Y`
    * `H`

* `apply_2_qubit_gate`:

        Applies a two qubit gate according to the specified control and target qubits. This will update the density
        matrix of the system accordingly.

            Parameters
            ----------
            gate : ndarray
                Array of dimension 2x2, examples are the well-known pauli matrices (X, Y, Z)
            cqubit : int
                Integer that indicates the control qubit. Note that the qubit counting starts at 0
            tqubit : int
                Integer that indicates the target qubit. Note that the qubit counting starts at 0.
            noise : bool, optional, default=None
                Determines if the gate is noisy. When the QuantumCircuit object is initialised
                with the 'noise' parameter to True, this parameter will also evaluate to True if
                not specified otherwise.
            pg : float [0-1], optional, default=None
                Specifies the amount of gate noise if present. If the QuantumCircuit object is
                initialised with a 'pg' parameter, this will be used if not specified otherwise
            draw : bool, optional, default=True
                If true, the specified gate will appear when the circuit is visualised.
            user_operation : bool, optional, default=True
                True if the user has requested the method and (else) False if it was invoked by an internal
                method.
    
    For which the known two-qubit gates are already provided:
    * `CNOT`
    * `CZ`

* `measure_first_N_qubits`:
            
        Method measures the first N qubits, given by the user, all in the 0 or 1 state.
        This will thus result in an even parity measurement. To also be able to enforce uneven
        parity measurements this should still be built!
        The density matrix of the system will be changed according to the measurement outcomes.

        *** MEASURED QUBITS WILL BE ERASED FROM THE SYSTEM AFTER MEASUREMENT, THIS WILL THUS
        DECREASE THE AMOUNT OF QUBITS IN THE SYSTEM WITH N ***

        Parameters
        ----------
        N : int
            Specifies the first n qubits that should be measured.
        measure : int [0 or 1], optional, default=0
            The measurement outcome for the qubits, either 0 or 1.
        uneven_parity : bool, optional, default=False
            If True, an uneven parity measurement outcome is forced on pairs of qubits.
        noise : bool, optional, default=None
             Whether or not the measurement contains noise.
        pm : float [0-1], optional, default=None
            The amount of measurement noise that is present (if noise is present).
        basis : str ["X" or "Z"], optional, default="X"
            Whether the measurement should be done in the X-basis or in the computational basis (Z-basis)
        user_operation : bool, optional, default=True
            True if the user has requested the method and (else) False if it was invoked by an internal
            method.

* `create_Bell_pairs_top`:

        This appends noisy Bell pairs on the top of the system. The noise is based on network noise
        modeled as (paper: https://www.nature.com/articles/ncomms2773.pdf)

            rho_raw = (1 - 4/3*pn) |psi><psi| + pn/3 * I,

        in which |psi> is a perfect Bell state.

        *** THIS METHOD APPENDS THE QUBITS TO THE TOP OF THE SYSTEM. THIS MEANS THAT THE AMOUNT OF
        QUBITS IN THE SYSTEM WILL GROW WITH '2N' ***

        Parameters
        ----------
        N : int
            Number of noisy Bell pairs that should be added to the top of the system.
        new_qubit: bool, optional, default=False
            If the creation of the Bell pair adds a new qubit to the drawing scheme (True) or reuses the top qubit
            (False) (this can be done in case the top qubit has been measured)
        noise : bool, optional, default=None
            Can be specified to force the creation of the Bell pairs noisy (True) or noiseless (False).
            If not specified (None), it will take the general noise parameter of the QuantumCircuit object.
        pn : float [0-1], optional, default=0.1
            The amount of network noise present
        user_operation : bool, optional, default=True
            True if the user has requested the method and (else) False if it was invoked by an internal
            method.

        Example
        -------
        qc.create_bell_pairs([(0, 1), (2, 3), (4,5)]) --> Creates Bell pairs between qubit 0 and 1,
        between qubit 2 and 3 and between qubit 4 and 5.

* `add_top_qubit`:

        Method appends a qubit with a given state to the top of the system.
        *** THE METHOD APPENDS A QUBIT, WHICH MEANS THAT THE AMOUNT OF QUBITS IN THE SYSTEM WILL
        GROW WITH 1 ***

        Parameters
        ----------
        qubit_state : array, optional, default=ket_0
            Qubit state, a normalised vector of dimension 2x1
        user_operation : bool, optional, default=True
            True if the user has requested the method and (else) False if it was invoked by an internal
            method.

* `set_qubit_states`:

        Sets the initial state of the specified qubits in the dict according to the specified state.

        *** METHOD SHOULD ONLY BE USED IN THE INITIALISATION PHASE OF THE CIRCUIT. SHOULD NOT BE USED
        AFTER OPERATIONS HAVE BEEN APPLIED TO THE CIRCUIT IN ORDER TO PREVENT ERRORS. ***

        Parameters
        ----------
        qubit_dict : dict
            Dictionary with the keys being the number of the qubits to be modified (first qubit is 0)
            and the value being the state the qubit should be in
        user_operation : bool, optional, default=True
            True if the user has requested the method and (else) False if it was invoked by an internal
            method.

        Example
        -------
        qc.set_qubit_state({0 : ket_1}) --> This sets the first qubit to the ket_1 state

* `draw_circuit`:

        Draws the circuit that corresponds to the operation that have been applied on the system,
        up until the moment of calling.

* `get_superoperator`:

        Returns the superoperator for the system. The superoperator is determined by taking the fidelities
        of the density matrix of the system [rho_real] and the density matrices obtained with any possible
        combination of error on the 4 data qubits in a noiseless version of the system
        [(ABCD) rho_ideal (ABCD)^]. Thus in equation form

        F[rho_real, (ABCD) * rho_ideal * (ABCD)^], {A, B, C, D} in {X, Y, Z, I}

        The fidelity is equal to the probability of this specific error, the combination of (ABCD), happening.

        Parameters
        __________
        qubits : list
            List of qubits of which the superoperator should be calculated. Only for these qubits it will be
            checked if certain errors occured on them. This is necessary to specify in case the circuit contains
            ancilla qubits that should not be evaluated. **The index of the qubits should be the index of the
            resulting density matrix, thus in case of measurements this can differ from the initial indices!!**
        proj_type : str, options: "X" or "Z"
            Specifies the type of stabilizer for which the superoperator should be calculated. This value is
            necessary for the postprocessing of the superoperator results if 'combine' is set to True.
        save_noiseless_density_matrix : bool, optional, default=True
            Whether or not the calculated noiseless (ideal) version of the circuit should be saved.
            This saved matrix will a next time be used for speedup if the same system is analysed with this method.
        combine : bool, optional, default=True
            Combines the error configuration on the data qubits that are equal up to permutation. This effectively
            means that for example [I, I, I, X] and [X, I, I, I] will be combined to one term [I, I, I, X] with the
            probabilities summed.
        most_likely : bool, optional, default=True
            Will choose the most likely configuration of degenerate configurations. This effectively means that the
            configuration with the highest amount of identity operators will be chosen. Only works if 'combine' is
            also set to True.
        print_to_console : bool, optional, default=True
            Whether the result should be printed in a clear overview to the console.
        file_name_noiseless : str, optional, default=None
            file name of the noiseless variant of the density matrix of the noisy system. Use this option if density
            matrix has been named manually and this one should be used for the calculations.
        file_name_measerror : str, optional, default=None
            file name of the noiseless variant with measurement error of the density matrix of the noisy system.
            Use this option if density matrix has been named manually and this one should be used for the
            calculations.
     
## Stabilizer measurement protocols for a distributed surface code

For the distributed surface code, this module contains the script `stabilizer_measurement_protocols.py` 
which simulates the stabilizer measurement protocols Stringent and Expedient that can be used to 
measure the stabilizers in a distributed surface code architecture. The file will calculate the superoperator for 
different values of network, gate and measurement noise.

To run the specific protocols, one can run the script from the root folder of the repository with the following
commandline arguments:

 * `-p`: specify the protocol(s), options: `monolithic`/`expedient`/`stringent`, default: `monolithic`
 * `-s`: specify stabilizer type, options: `Z`/`X`, default: `Z`
 * `-pg`: specify the gate error probability/probabilities, [float 0-1], default=`0.006`
 * `-pm`: specify the measurement error probability/probabilities, [float 0-1], default=`0.006`
 * `-pn`: specify the network error probability/probabilities, [float 0-1], default=`0.1`
 * `--pm_equals_pg`: add this flag if it holds that `pm=pg`
 * `-tr`: add this flag if the runs should be run in parallel
 * `-sv`: use if csv file(s) of the superoperator(s) should be saved, optional
 * `-fn`: specify the filename/filenames of the superoperator, optional
 * `-c`: use if the output to the console should contain color for clearness, optional
 * `-ltsv`: use if a pdf file of the noisy circuit drawn in LaTeX should be save to the 'circuit_pdfs' folder.
 * `-pr`: add if only the order of the runs should be printed. This can be useful for debugging or filenaming purposes

So the command looks something like
```
python circuit_simulation/stabilizer_measurement_protocols.py -p monolithic expedient -s Z -pg 0.00 65 0.007 0.0075 -pm 0.0075 -pn 0.11
```

Note: When saving the superoperator csv files with manual naming (`-fn`), one can first add `-pr` to the command to only
get the order of the runs. From this, the naming order for the `-fn` flag can be deduced.