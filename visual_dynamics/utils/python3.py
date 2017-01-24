import inspect
import sys


def get_kwargs(default_kwargs, kwargs):
    for key in kwargs.keys():
        if key not in default_kwargs:
            fn_name = inspect.stack()[1][3]
            if not kwargs:
                raise TypeError('%r got an unexpected keyword argument %s' % (fn_name, list(kwargs.keys())[0]))
    updated_kwargs = dict(default_kwargs)
    updated_kwargs.update(kwargs)
    return updated_kwargs


def get_signature_args(f):
    if sys.version_info.major == 2:
        argspec = inspect.getargspec(f)
        args = argspec.args
    elif sys.version_info.major == 3:
        sig = inspect.signature(f)
        args = list(sig.parameters.keys())
    else:
        raise ValueError('Unknown python version %d', sys.version_info.major)
    return args
