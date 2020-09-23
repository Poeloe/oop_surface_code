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
