'''
2020 Mark Shui Hu, QuTech

www.github.com/watermarkhu/oop_surface_code
_____________________________________________


We define the unit cell, which contains two qubits, a star operator and plaquette operator.

    |       |
- Star  -  Q_0 -     also top (T) qubit
    |       |
-  Q_1  - Plaq  -    also down (D) qubit
    |       |

Each cell is indicated by its y and x coordiantes. As such every qubit and stabilizer can by identified by a unique ID number:

Qubits: qID (td, y, x)          Stabilizers: sID (ertype, y, x)
    Q_0:    (0, y, x)               Star:   (0, y, x)
    Q_1:    (1, y, x)               Plaq:   (1, y, x)

The 3D graph (toric/planar) is a cubic lattice with many layer of these unit cells.

'''

from . import graph_2D as go
from ..plot import plot_graph_lattice as pgl
from ..plot import plot_unionfind as puf
import random

class toric(go.toric):
    '''
    Inherits all the class variables and methods of graph_2D.toric

    Additions:
        G   dict of qubit-like objects called bridges containing the vertical edges connecting stabs of different layers
                Key:    sID number
                Value:  Stab object
    Replaces:
        init_uf_plot()
        apply_and_measure_errors()
        init_erasure()
        init_paul()
        measure_stab()
        logical_error()
        count_matching_weight()
        reset()

    3D graph is initilized by calling the init_graph_layer() method of the parent graph_2D.toric object.
    From here, we call that method size-1 times again, on each layer of the cubic lattice. Furthermore, qubit-like objects bridges containing vertical edges are added between the layers.
    Dim dimension is set to 3 and decoder_layer is set to last layer.
    '''

    def __init__(self, size, decoder, plot_config={}, dim=3, cycles=None, *args, **kwargs):

        plot2D = kwargs.pop("plot2D", 0)
        super().__init__(size, decoder, *args, plot2D=0, dim=3, **kwargs)

        self.cycles = cycles if cycles is not None else size
        self.decode_layer = self.cycles - 1
        self.dim = 3
        self.G = {}
        self.GHZ_failed = {}

        for z in range(1, self.cycles):
            self.init_graph_layer(z=z)
            self.G[z] = {}

            for vU, vD in zip(self.S[z].values(), self.S[z-1].values()):
                bridge = self.G[z][vU.sID] = Bridge(qID=vU.sID, z=z)

                vU.neighbors["d"] = (vD, bridge.E)
                vD.neighbors["u"] = (vU, bridge.E)


        for key, value in kwargs.items():
            setattr(self, key, value)
        self.plot2D = plot2D
        self.plot_config = plot_config
        self.gl_plot = pgl.plot_3D(self, **plot_config) if self.plot3D else None


    def init_uf_plot(self):
        self.uf_plot = puf.plot_3D(self, **self.plot_config)
        return self.uf_plot


    def __repr__(self):
        return f"3D {self.__class__.__name__} graph object"


    def count_matching_weight(self):
        '''
        Applies count_matching_weight() method of parent graph_2D object on each layer of the cubic lattice. Additionally counts the weight of the edges in the bridge objects present in the 3D graph.
        '''
        weight = 0
        for z in range(self.cycles):
            for qubit in self.Q[z].values():
                if qubit.E[0].matching == 1:
                    weight += 1
                if qubit.E[1].matching == 1:
                    weight += 1
        for layer in self.G.values():
            for bridge in layer.values():
                if bridge.E.matching:
                    weight += 1
        self.matching_weight.append(weight)

    '''
    ########################################################################################

                                    Surface code methods

    ########################################################################################
    '''


    def apply_and_measure_errors(self, pX, pZ, pE, pmX, pmZ, **kwargs):
        '''
        Initilizes errors on the qubits and measures the stabilizers on the graph on each layer of the cubic lattice.
        For the first size-1 layers, measurement errors are applied.
        For the final layer, perfect measurements are applied to ensure anyon creation.
        '''

        # first layers initilized with measurement error
        for z in range(self.cycles)[:-1]:
            self.init_erasure(pE=pE, z=z)
            self.init_pauli(pX=pX, pZ=pZ, pE=pE, z=z, set_prev_value=True)
            self.measure_stab(pmX=pmX, pmZ=pmZ, z=z)

        # final layer initialized with perfect measurements
        self.set_qubit_states_to_state_previous_layer(z=self.decode_layer)
        self.measure_stab(z=self.decode_layer)

        if self.gl_plot:
            if pE != 0:
                for z in range(self.cycles):
                    self.gl_plot.plot_erasures(z, draw=False)
                self.gl_plot.draw_plot()
            for z in range(self.cycles):
                self.gl_plot.plot_errors(z, draw=False)
            self.gl_plot.draw_plot()
            for z in range(self.cycles):
                self.gl_plot.plot_syndrome(z)
                self.gl_plot.draw_plot()

    def perform_stabilizer_measurement_cycles_with_superoperator(self, superoperator, networked_architecture=False):
        """
            Method appoints the superoperator object to the superoperator attribute of the graph object. With this
            superoperator it invokes another method to apply qubit error and measurement errors for every 'z' layer.

            Parameters
            ----------
            superoperator : Superoperator object
                A Superoperator object that will be used to apply qubit and measurement error to the system.
            networked_architecture : bool, optional, default=False
                If True, stabilizers measurements will be handled such that it mimics the situation for a surface code
                with a networked architecture.
        """
        self.superoperator = superoperator
        self.superoperator.set_stabilizer_rounds(self)
        for z in range(self.cycles)[:-1]:
            if not networked_architecture:
                self.stabilizer_cycle_monolithic_architecture(z)
            elif self.superoperator.sup_op_elements_idle and self.superoperator.additional_superoperators:
                self.stabilizer_cycle_three_superoperators(z)
            elif self.superoperator.sup_op_elements_idle:
                pass
            else:
                self.stabilizer_cycle_with_superoperator(z)

        # For decoder layer get the qubit state of the previous layer and measure perfectly
        self.set_qubit_states_to_state_previous_layer(z=self.decode_layer)
        self.measure_stab(z=self.decode_layer)

        if self.gl_plot:
            for z in range(self.cycles):
                self.gl_plot.plot_erasures(z, draw=False)
            self.gl_plot.draw_plot()
            for z in range(self.cycles):
                self.gl_plot.plot_errors(z, draw=False)
            self.gl_plot.draw_plot()
            for z in range(self.cycles):
                self.gl_plot.plot_syndrome(z)
                self.gl_plot.draw_plot()

    def post_process_failed_stabilizers(self):
        if self.size < 4:
            return
        for sID, failed_stabs in self.GHZ_failed.items():
            skip_layers = [stab.z for stab in failed_stabs]
            parities = [self.S[layer][sID].parity for layer in range(self.cycles) if (layer not in skip_layers)]
            parity = 1 if parities.count(1) > (int(len(parities)/2)) else 0

            for failed_stab in failed_stabs:
                failed_stab.parity = parity

    def post_process_failed_stabilizers_lower_upper(self):
        for sID, failed_stabs in self.GHZ_failed.items():
            skip_layers = [stab.z for stab in failed_stabs]
            for stab in failed_stabs:
                parities = [self.S[layer][sID].parity if layer != -1 else 0 for layer in [stab.z - 1, stab.z + 1] if
                            (layer not in skip_layers)]
                stab.parity = 1 if parities.count(1) > int(len(parities)/2) else 0
                skip_layers.remove(stab.z)

    def set_qubit_states_to_state_previous_layer(self, z):
        if z == 0:
            return
        for qubit in self.Q[z-1].values():
            if not qubit.E[0].state and not qubit.E[1].state:
                continue
            if qubit.E[0].state:
                self.Q[z][qubit.qID].E[0].state = 1
            if qubit.E[1].state:
                self.Q[z][qubit.qID].E[1].state = 1

    def init_erasure(self, pE=0, z=0, **kwargs):
        """
        Initializes an erasure error with probability pE, which will take form as a uniformly chosen pauli X and/or Z error.
        Qubit states from previous layer are copied to this layer, whereafter erasure error is applied.
        """

        if pE == 0:
            return

        for qubitu in self.Q[z].values():

            # Get qubit state from previous layer
            if z != 0:
                qubitu.E[0].state, qubitu.E[1].state = (self.Q[z-1][qubitu.qID[:3]].E[n].state for n in range(2))

            # Apply errors
            if random.random() < pE:
                qubitu.erasure = 1
                rand = random.random()
                if rand < 0.25:
                    qubitu.E[0].state = 1 - qubitu.E[0].state
                elif rand >= 0.25 and rand < 0.5:
                    qubitu.E[1].state = 1 - qubitu.E[1].state
                elif rand >= 0.5 and rand < 0.75:
                    qubitu.E[0].state = 1 - qubitu.E[0].state
                    qubitu.E[1].state = 1 - qubitu.E[1].state


    def init_pauli(self, pX=0, pZ=0, pE=0, z=0, set_prev_value=False, **kwargs):
        """
        initiates Pauli X and Z errors on the lattice based on the error rates
        Qubit states from previous layer are copied to this layer, whereafter pauli error is applied.
        """

        if pX == 0 and pZ == 0:
            return

        for qubitu in self.Q[z].values():

            # Get qubit state from previous layer if not aleady done
            if pE == 0 and z != 0 and set_prev_value:
                qubitu.E[0].state, qubitu.E[1].state = (self.Q[z-1][qubitu.qID].E[n].state for n in [0, 1])

            # Apply errors
            if pX != 0 and random.random() < pX:
                qubitu.E[0].state = 1 - qubitu.E[0].state
            if pZ != 0 and random.random() < pZ:
                qubitu.E[1].state = 1 - qubitu.E[1].state


    def measure_stab(self, pmX=0, pmZ=0, z=0, GHZ_success=1, stabs=None, measurement_errors=None, **kwargs):
        """
            Method measures the stabilizers and registers the outcome to the stabilizer object itself. After, it checks
            if the stabilizer should be saved as an anyon by comparing its parity value to the parity value in the
            previous layer

            Parameters
            ----------
            pmX : float
                Float value to indicate the rate of X errors, used as measurement error rate for star stabilizers.
            pmZ : float
                Float value to indicate the rate of Z errors, used as measurement error rate for plaquette stabilizers.
            z : int
                Indicates the layer of which the stabilizers should be calculated.
            GHZ_success : float, optional, default=1
                Indicates the amount of GHZ stabilizers that have been completed.
            stabs : list, optional, default=None
                List containing the (subset of) stabilizers that should be measured.
            measurement_errors : list, optional, default=None
                List containing boolean values that indicate whether or not a measurement error has occurred. The
                indices of the list should correspond with the indices of the stabilizers.
        """
        if stabs is None:
            stabs = self.S[z].values()

        for i, stab in enumerate(stabs):

            # If GHZ state is malformed measurement result will be the result of previous layer and rest will be skipped
            if random.random() > GHZ_success:
                if stab.sID in self.GHZ_failed:
                    self.GHZ_failed[stab.sID].append(stab)
                else:
                    self.GHZ_failed[stab.sID] = [stab]
                stab.parity = 0 if z == 0 else self.S[z-1][stab.sID].parity
                continue

            # Get parity of stabilizer
            stab.parity = 0
            for dir in self.dirs:
                if dir in stab.neighbors:
                    _, edge = stab.neighbors[dir]
                    if edge.state:
                        stab.parity = 1 - stab.parity

            # Apply measurement error unless the layer is equal to the decode layer
            if z != self.decode_layer:
                pM = pmX if stab.sID[0] == 0 else pmZ
                if (pM != 0 and random.random() < pM) or (measurement_errors is not None and measurement_errors[i]):
                    stab.parity = 1 - stab.parity
                    stab.mstate = 1

            # Save vertex as anyon if parity different than previous layer
            stabd_state = 0 if z == 0 else self.S[z-1][stab.sID].parity
            stab.state = 0 if stabd_state == stab.parity else 1


    def logical_error(self):
        '''
        Applies logical_error() method of parent graph_2D object on the last layer.
        '''
        if self.plot2D:
            self.gl2_plot = pgl.plot_2D(self, z=self.decode_layer, from3D=1, **self.plot_config)
            self.gl2_plot.new_iter("Final layer errors")
            self.gl2_plot.plot_errors(z=self.decode_layer, draw=1)
        # 22-07-2020: Fixed bug that caused program to evaluate the wrong layer for logical errors
        return super().logical_error(z=self.decode_layer)

    '''
    ########################################################################################

                                    Constructor methods

    ########################################################################################
    '''

    def reset(self):
        '''
        Applies reset() method of parent graph_2D object. Also resets all the bridge objects present in the 3D graph.
        '''
        super().reset()
        self.GHZ_failed = {}
        for layer in self.G.values():
            for bridge in layer.values():
                bridge.reset()

'''
########################################################################################

                                        Planar class

########################################################################################
'''

class planar(toric, go.planar):
    '''
    Inherits all the calss variables and methods of graph_3D.toric and graph_3D.planar.

    graph_3D.planar -> graph_3D.toric -> graph_2D.planar -> graph_2D.toric

    All super().def() methods in graph_3D.toric now call on graph_2D.planar, such that the planar structure is preserved.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

'''
########################################################################################

                            Subclasses: Graph objects

########################################################################################
'''

class Bridge(object):
    '''
    Qubit-like object that contains a single vertical Edge object that connects Stabs between different layers in the cubic lattice.

    qID         (td, y, x)
    z           layer of graph to which this stab belongs
    E           list countaining the two edges of the primal and secundary lattice
    erasure     placeholder to ensure decoder works
    '''
    def __init__(self, qID, z=0):
        self.qID = qID       # (ertype, y, x)
        self.z = z
        self.erasure = 0
        self.E = go.Edge(self, ertype=qID[0], edge_type=1, z=z)

    def __repr__(self):
        errortype = "X" if self.qID[0] == 0 else "Z"
        return "g{}({},{}:{})".format(errortype, *self.qID[1:], self.z)

    def picker(self):
        return self.__repr__()

    def reset(self):
        """
        Changes all iteration parameters to their initial value
        """
        self.E.reset()
