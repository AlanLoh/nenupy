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
def accepts(*types):
    """ Decorator
    """

    def decorator(func):
        # Check that all the types are set
        assert len(types) == func.__code__.co_argcount,\
            'Number of types does not match argument number.'
        argnames = func.__code__.co_varnames
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            newKwargs = {}
            newArgs = []
            for (arg, typ) in zip(argnames, types):
                if arg in kwargs.keys():
                    # Check if kwargs have been filled
                    if not isinstance(kwargs[arg], typ):
                        if isinstance(typ, tuple):
                            if len(typ) > 1:
                                typ = typ[0] 
                        try:
                            # Try to convert to correct type 
                            kwargs[arg] = typ(kwargs[arg])
                        except:
                            raise TypeError(
                                f'`{arg}` should be a `{typ}` object.'
                            )
                    newKwargs[arg] = kwargs[arg]
                else:
                    # Check if args have been filled
                    index = argnames.index(arg)
                    if index >= len(args):
                        # Default value written in the function
                        continue
                    if not isinstance(args[index], typ):
                        if isinstance(typ, tuple):
                            if len(typ) > 1:
                                typ = typ[0] 
                        try:
                            # Try to convert to correct type 
                            newArg = typ(args[index])
                        except:
                            raise TypeError(
                                f'`{args[index]}` should be a `{typ}` object.'
                            )
                    else:
                        newArg = args[index]
                    newArgs.append(newArg)
            return func(*newArgs, **newKwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator

# ============================================================= #

