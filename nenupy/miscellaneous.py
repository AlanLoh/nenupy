#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    **************************************
    Miscellaneous functions and decorators
    **************************************
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'check_type',
    'time_args'
]


from functools import wraps
import inspect
from astropy.time import Time

# def wrapped_decorator(func):
#     """wrapped decorator docstring"""
#     @wraps(func)
#     def inner_function(*args, **kwargs):
#         """inner function docstring """
#         print func.__name__ + "was called"
#         return func(*args, **kwargs)
#     return inner_function


# def decorator(arg1, arg2):
#     def inner_function(function):
#         @wraps(function)
#         def wrapper(*args, **kwargs):
#             print "Arguements passed to decorator %s and %s" % (arg1, arg2)
#             function(*args, **kwargs)
#         return wrapper
#     return inner_function

# ============================================================= #
# ------------------------ check_type ------------------------- #
# ============================================================= #
def check_type(variable, value, typeof):
    """
    """
    if not isinstance(value, typeof):
        raise TypeError(
            f'`{variable}` should be a `{typeof}` object.'
        )
# ============================================================= #


# ============================================================= #
# ------------------------- time_args ------------------------- #
# ============================================================= #
def time_args(*t_args):
    """ Decorator to check for a correct time argument.
    """

    def inner_function(func):
        """ Inner function that needd to be returned after check
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            arg_names = inspect.getfullargspec(func).args
            for t_arg in t_args:
                if t_arg in arg_names:
                    # Check if kwargs have been filled
                    if t_arg in kwargs.keys():
                        check_type(t_arg, kwargs[t_arg], Time)

                    # Check if args have been filled
                    else:
                        index = arg_names.index(t_arg)
                        check_type(t_arg, args[index], Time)

                else:
                    raise ValueError(
                        f'{t_arg} not in {arg_names}!'
                    )

            return func(*args, **kwargs)

        return wrapper

    return inner_function
# ============================================================= #

