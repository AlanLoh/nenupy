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
    'accepts'
]


import functools


# ============================================================= #
# -------------------------- accepts -------------------------- #
# ============================================================= #
def accepts(*types, strict=True):
    """ Decorator
    """

    def decorator(func, strict=strict):
        # Check that all the types are set
        assert len(types) == func.__code__.co_argcount,\
            'Number of types does not match argument number.'
        argnames = func.__code__.co_varnames
        if not isinstance(strict, tuple):
            strict = (strict, )*func.__code__.co_argcount
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Make everything kwargs, including defaults
            newKwargs = {}
            nFilled = len(args) + len(kwargs)
            for i in range(func.__code__.co_argcount):
                if i < len(args):
                    newKwargs[argnames[i]] = args[i]
                elif i >= nFilled:
                    newKwargs[argnames[i]] = func.__defaults__[i - nFilled]
                else:
                    newKwargs[argnames[i]] = kwargs[argnames[i]]

            for index, (arg, typ) in enumerate(zip(argnames, types)):
                # Check if kwargs have been filled
                if not isinstance(newKwargs[arg], typ):
                    if strict[index]:
                        raise TypeError(
                            f'`{arg}` of type `{type(newKwargs[arg])}` should be a `{typ}` object.'
                        )
                    if isinstance(typ, tuple):
                        if len(typ) > 1:
                            typ = typ[0] 
                    try:
                        # Try to convert to correct type 
                        newKwargs[arg] = typ(newKwargs[arg])
                    except:
                        raise TypeError(
                            f'`{arg}` of type `{type(newKwargs[arg])}` should be a `{typ}` object.'
                        )

            return func(**newKwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator

# ============================================================= #

