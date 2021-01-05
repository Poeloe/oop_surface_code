import inspect
import functools


def handle_none_parameters(func=None, *, excluded_parameters=None):
    """
        Decorator is used to set parameters with default=None that are not specified by the user to the according
        attribute present in the object.

        Example:
            Say an object 'QuantumCircuit' has a boolean attribute 'noise' which registers if the QuantumCircuit in
            general experiences noise (yes if set to True). Now lets say the method 'X(qubit, noise=None)' inside the
            QuantumCircuit class (that applies an X-gate) has this same 'noise' parameter. When not specified by the
            user (so default value 'None' is used) this should get the same value as present for the 'noise' attribute
            in the QuantumCircuit object. This decorator handles this last step.
    """
    if not func:
        return functools.partial(handle_none_parameters, excluded_parameters=excluded_parameters)

    @functools.wraps(func)
    def set_nones_to_object_value(*args, **kwargs):
        nonlocal excluded_parameters
        excluded_parameters = [] if excluded_parameters is None else excluded_parameters
        parameter_names = [p.name for p in inspect.signature(func).parameters.values() if
                           (p.name not in kwargs.keys() or kwargs[p.name] is None) and p.default is None]
        parameter_names = list(set(parameter_names).difference(set(excluded_parameters)))
        for name in parameter_names:
            kwargs[name] = getattr(args[0], name)
        return func(*args, **kwargs)
    return set_nones_to_object_value


def determine_qubit_index(func=None, parameter_positions=None):
    """
        Decorator used to determine qubit indices based on a the passed string

        Parameters
        ----------
        parameter_positions : list
            Positions of the parameters in the method signature that should be checked for qubit_strings. Counting
            starts at 0.
    """
    if not func:
        return functools.partial(determine_qubit_index, parameter_positions=parameter_positions)

    @functools.wraps(func)
    def get_qubit_index(*args, **kwargs):
        nonlocal parameter_positions
        qc = args[0]

        # First handle the kwargs
        parameters_list = ['tqubit', 'cqubit', 'qubit1', 'qubit2', 'measure_qubits']
        qubit_strings = [kwargs[i] for i in parameters_list if i in kwargs and type(kwargs[i]) in [str, list]]
        if 'measure_qubits' in kwargs:
            [qubit_strings.append(qubit) for qubit in kwargs['measure_qubits'] if type(qubit) == str]
        for qubit_string, parameter in zip(qubit_strings, parameters_list):
            if type(qubit_string) == list:
                qubit_index = [qc._operations.gate_operations.determine_node_qubit_from_string(args[0], qubit)
                               for qubit in qubit_string if type(qubit) == str]
            else:
                qubit_index = qc._operations.gate_operations.determine_node_qubit_from_string(args[0], qubit_string)
            kwargs[parameter] = qubit_index if qubit_index is not None else kwargs[parameter]

        # Handle the args
        list_args = list(args)
        for i, arg in enumerate(list_args):
            if i in parameter_positions:
                if type(arg) == str:
                    list_args[i] = qc._operations.gate_operations.determine_node_qubit_from_string(args[0], arg)
                # If arg is a list, all members should be translated to qubit indices
                if type(arg) == list:
                    qubit_indices = []
                    for qubit in arg:
                        if type(qubit) == str:
                            qubit_indices.append(qc._operations.gate_operations.determine_node_qubit_from_string(
                                args[0], qubit))
                    list_args[i] = qubit_indices if qubit_indices else list_args[i]

        return func(*tuple(list_args), **kwargs)
    return get_qubit_index


def skip_if_cut_off_reached(func=None, *, run_once=False):
    """
        Decorator which is used to decorate QuantumCircuit methods that should be skipped when circuit cut-off time is
        reached. If a method should still be ran once before skipping, this can be indicated with the run_once parameter

        Parameters
        ----------
        run_once : bool
            Indicated whether the decorated method should be performed one last time in case of skipping.
    """
    run_once_funcs = {}
    if not func:
        return functools.partial(skip_if_cut_off_reached, run_once=run_once)

    def should_run_once(self):
        nonlocal run_once_funcs
        nonlocal run_once
        old_value = SKIP()
        if run_once:
            if func.__name__ not in run_once_funcs:
                run_once_funcs[func.__name__] = 1
                old_value = self._circuit_operations_ended
                # Set _circuit_operations_ended to True, since everything inside the passed function should run.
                # After function is finished the value will be returned to the original value
                self._circuit_operations_ended = True
            else:
                run_once_funcs[func.__name__] += 1
        return old_value

    @functools.wraps(func)
    def determine_skip(*args, **kwargs):
        nonlocal run_once_funcs
        nonlocal run_once
        old_value = SKIP()
        self = args[0]

        # When circuit operations ended, methods should no longer be skipped. Skipping only holds for circuit operations
        if self._circuit_operations_ended:
            retval = func(*args, **kwargs)

        # If the cut-off time of the QuantumCircuit object is reached, circuit operations must be skipped
        elif self.cut_off_time_reached:
            old_value = should_run_once(self)
            retval = SKIP() if old_value == SKIP() else func(*args, **kwargs)

        # If the cut-off time of the QuantumCircuit object is NOT reached, sub circuits should still run till cut-off
        # of the sub circuit itself is reached
        elif self._current_sub_circuit is not None and self._current_sub_circuit.cut_off_time_reached:
            old_value = should_run_once(self)
            retval = SKIP() if old_value == SKIP() else func(*args, **kwargs)

        # If nothing holds, the function should be ran as usual
        else:
            retval = func(*args, **kwargs)

        if run_once and old_value != SKIP():
            self._circuit_operations_ended = old_value

        return retval
    return determine_skip


class SKIP:
    """ Class is used as an return value, such that when cut off is reached the code will not end up in an infinite
        loop due to the Failure Reset Levels off the protocols (if one returns None, the while loops will never get
        the expected True value)"""

    def __init__(self):
        self.name = "SKIP"

    def __eq__(self, other):
        if type(other) != SKIP:
            return True
        return self.name == other.name
