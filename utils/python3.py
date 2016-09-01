import inspect


def get_kwargs(default_kwargs, kwargs):
    for key in kwargs.keys():
        if key not in default_kwargs:
            fn_name = inspect.stack()[1][3]
            if not kwargs:
                raise TypeError('%r got an unexpected keyword argument %s' % (fn_name, list(kwargs.keys())[0]))
    updated_kwargs = dict(default_kwargs)
    updated_kwargs.update(kwargs)
    return updated_kwargs
