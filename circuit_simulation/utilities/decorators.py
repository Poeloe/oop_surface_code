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
        old_value = None
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
        old_value = None
        self = args[0]

        # When circuit operations ended, methods should no longer be skipped. Skipping only holds for circuit operations
        if self._circuit_operations_ended:
            retval = func(*args, **kwargs)

        # If the cut-off time of the QuantumCircuit object is reached, circuit operations must be skipped
        elif self.cut_off_time_reached:
            old_value = should_run_once(self)
            retval = None if old_value is None else func(*args, **kwargs)

        # If the cut-off time of the QuantumCircuit object is NOT reached, sub circuits should still run till cut-off
        # of the sub circuit itself is reached
        elif self._current_sub_circuit is not None and self._current_sub_circuit.cut_off_time_reached:
            old_value = should_run_once(self)
            retval = None if old_value is None else func(*args, **kwargs)

        # If nothing holds, the function should be ran as usual
        else:
            retval = func(*args, **kwargs)

        if run_once and old_value is not None:
            self._circuit_operations_ended = old_value

        return retval
    return determine_skip

