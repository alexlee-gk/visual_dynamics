import sys

import numpy as np
import yaml
from yaml.composer import Composer
from yaml.constructor import Constructor
from yaml.parser import Parser
from yaml.reader import Reader
from yaml.resolver import Resolver
from yaml.scanner import Scanner

from visual_dynamics.utils.python3 import get_signature_args


class Python2to3Constructor(Constructor):

    def find_python_name(self, name, mark):
        if sys.version_info.major == 3:
            if name == '__builtin__.dict':
                name = 'builtins.dict'
        return Constructor.find_python_name(self, name, mark)


class Python2to3Loader(Reader, Scanner, Parser, Composer, Python2to3Constructor, Resolver):

    def __init__(self, stream):
        Reader.__init__(self, stream)
        Scanner.__init__(self)
        Parser.__init__(self)
        Composer.__init__(self)
        Python2to3Constructor.__init__(self)
        Resolver.__init__(self)


def get_config(instance):
    if isinstance(instance, ConfigObject):
        config = instance.get_config()
    elif isinstance(instance, dict):
        config = {k: get_config(v) for k, v in instance.items()}
        if 'class' in config:
            config['__class__'] = dict
    elif isinstance(instance, (tuple, list, set)):
        config = type(instance)(get_config(v) for v in instance)
    else:
        config = instance
    return config


def from_config(config, replace_config=None):
    if isinstance(config, dict):
        if '__class__' in config:
            cls = config.get('__class__')
        else:
            cls = config.get('class', dict)
        if issubclass(cls, ConfigObject):
            instance = cls.from_config(config, replace_config=replace_config)
        else:
            config = dict(config)  # shallow copy
            if '__class__' in config:
                config.pop('__class__')
            else:
                config.pop('class', None)
            instance = {k: from_config(v, replace_config=replace_config) for k, v in config.items()}
    elif isinstance(config, (tuple, list, set)):
        instance = type(config)(from_config(v, replace_config=replace_config) for v in config)
    else:
        instance = config
    return instance


def to_yaml(instance, *args, **kwargs):
    config = get_config(instance)
    return yaml.dump(config, *args, width=float('inf'), **kwargs)


def from_yaml(yaml_string):
    config = yaml.load(yaml_string, Loader=Python2to3Loader)
    return from_config(config)


class ConfigObject(object):
    """
    All config dictionaries should have an entry for 'class' and optionally an
    entry for '__class__'. The value of either entries are used to instantiate
    an object of that class, with the value for '__class__' (if specified)
    overriding the value for 'class'.
    """
    def __repr__(self, config=None):
        class_name = self.__class__.__name__
        config = config or self._get_config()
        # attributes (including properties) that are in config
        attr_dict = {k: getattr(self, k) for k in dir(self) if k in config}
        # order attributes based in the order in which they appear in the constructor
        ordered_attr_pairs = []
        for arg_name in get_signature_args(self.__init__):
            try:
                ordered_attr_pairs.append((arg_name, attr_dict.pop(arg_name)))
            except KeyError:
                pass
        # add remaining attributes that doesn't appear in the constructor
        ordered_attr_pairs += list(attr_dict.items())
        kwargs = ', '.join(['%s=%r' % (k, v) for (k, v) in ordered_attr_pairs])
        return "%s(%s)" % (class_name, kwargs)

    @staticmethod
    def convert_array_tolist(config):
        for k, v in config.items():
            if isinstance(v, np.ndarray):
                config[k] = v.tolist()

    def _get_config(self):
        """
        This method should return a config without the class in it.
        """
        return {}

    def get_config(self):
        """
        This method shouldn't need to be overridden. Use _get_config instead
        for doing that.
        This method should return a config with the class in it.
        """
        config = get_config(self._get_config())
        config['class'] = self.__class__
        return config

    @classmethod
    def _from_config(cls, config):
        """
        The config should not have the class in it.
        """
        try:
            return cls(**config)
        except TypeError as e:
            print(e)
            import IPython as ipy; ipy.embed()

    @classmethod
    def from_config(cls, config, replace_config=None):
        """
        The config should have the class in it.
        """
        config = dict(config)  # shallow copy
        if '__class__' in config:
            cls_other = config.pop('__class__')
        else:
            cls_other = config.pop('class')
        if cls != cls_other:
            raise ValueError('this class %r is different from class in config %r' % (cls, cls_other))
        if replace_config:
            for k in config.keys():
                if k in replace_config:
                    config[k] = replace_config[k]
        return cls._from_config(from_config(config, replace_config=replace_config))

    def to_yaml(self, *args, **kwargs):
        return to_yaml(self, *args, **kwargs)

    @classmethod
    def from_yaml(cls, yaml_string):
        instance = from_yaml(yaml_string)
        cls_other = instance.__class__
        if cls != cls_other:
            raise ValueError('this class %r is different from class in config %r' % (cls, cls_other))
        return instance

    def __getstate__(self):
        return self.get_config()

    def __setstate__(self, d):
        instance = from_config(d)
        self.__dict__.update(instance.__dict__)
