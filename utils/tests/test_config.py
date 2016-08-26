from nose2 import tools
from utils import ConfigObject, get_config, from_config


class Number(ConfigObject):
    def __init__(self, one, two, number_config=None):
        self.one = one
        self.two = two
        self.number_config = number_config
        self.other = None

    def _get_config(self):
        config = super(Number, self)._get_config()
        config.update({'one': self.one,
                       'two': self.two,
                       'number_config': self.number_config})
        return config

    def __eq__(self, other):
        return isinstance(other, Number) and self._get_config() == other._get_config()

    def __ne__(self, other):
        return not self.__eq__(other)


@tools.params(Number(-1, -2),
              Number(1, 2, number_config=Number(-1, -2).get_config()),
              Number(1, 2, number_config=Number(-1, -2))
              )
def test_same_function_method_get_config(number):
    assert number.get_config() == get_config(number)


@tools.params(Number(-1, -2),
              Number(1, 2, number_config=Number(-1, -2).get_config()),
              Number(1, 2, number_config=Number(-1, -2))
              )
def test_roundtripping_function(number):
    config = get_config(number)
    assert from_config(config) == number
    assert get_config(from_config(config)) == config


@tools.params(Number(-1, -2),
              Number(1, 2, number_config=Number(-1, -2))
              )
def test_deep_roundtripping_function(number):
    assert from_config(from_config(get_config(get_config(number)))) == number
    assert from_config(get_config(get_config(number))) != number


@tools.params(Number(-1, -2),
              Number(1, 2, number_config=Number(-1, -2).get_config()),
              Number(1, 2, number_config=Number(-1, -2))
              )
def test_roundtripping_private_method(number):
    config = number._get_config()
    assert number._from_config(config) == number
    assert number._from_config(config)._get_config() == config


def test_neg_number_get_config():
    neg_number = Number(-1, -2)
    config = {'class': Number, 'one': -1, 'two': -2, 'number_config': None}
    assert neg_number.get_config() == config
    assert get_config(neg_number) == config


def test_number_c_get_config():
    neg_number = Number(-1, -2)
    number_c = Number(1, 2, number_config=neg_number.get_config())
    config = {'class': Number,
              'one': 1,
              'two': 2,
              'number_config': {'__class__': dict,
                                'class': Number,
                                'one': -1,
                                'two': -2,
                                'number_config': None}}
    assert number_c.get_config() == config
    assert get_config(number_c) == config


def test_number_get_config():
    neg_number = Number(-1, -2)
    number = Number(1, 2, number_config=neg_number)
    config = {'class': Number,
              'one': 1,
              'two': 2,
              'number_config': {'class': Number,
                                'one': -1,
                                'two': -2,
                                'number_config': None}}
    assert number.get_config() == config
    assert get_config(number) == config


def test_neg_number_from_config():
    neg_number = Number(-1, -2)
    config = {'class': Number, 'one': -1, 'two': -2, 'number_config': None}
    assert from_config(config) == neg_number
    assert neg_number.from_config(config) == neg_number


def test_number_c_from_config():
    neg_number = Number(-1, -2)
    number_c = Number(1, 2, number_config=neg_number.get_config())
    config = {'class': Number,
              'one': 1,
              'two': 2,
              'number_config': {'__class__': dict,
                                'class': Number,
                                'one': -1,
                                'two': -2,
                                'number_config': None}}
    assert from_config(config) == number_c
    assert number_c.from_config(config) == number_c


def test_number_from_config():
    neg_number = Number(-1, -2)
    number = Number(1, 2, number_config=neg_number)
    config = {'class': Number,
              'one': 1,
              'two': 2,
              'number_config': {'class': Number,
                                'one': -1,
                                'two': -2,
                                'number_config': None}}
    assert from_config(config) == number
    assert number.from_config(config) == number
