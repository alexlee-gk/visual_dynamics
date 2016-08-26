import numpy as np
from nose2 import tools
import utils
from utils.transformer import Transformer, OpsTransformer, ImageTransformer, CompositionTransformer


@tools.params(Transformer(),
              OpsTransformer(scale=2.0, offset=-1.0),
              OpsTransformer(scale=2.0 / 255.0, offset=-1.0, exponent=-1, transpose=(2, 0, 1)),
              ImageTransformer(scale_size=0.125, crop_size=[32, 32], crop_offset=[0, 0]),
              CompositionTransformer([]),
              CompositionTransformer(
                  [OpsTransformer(scale=2.0 / 255.0, offset=-1.0, transpose=(2, 0, 1)),
                   ImageTransformer(scale_size=0.125, crop_size=[32, 32], crop_offset=[0, 0])]),
              )
def test_roundtripping(transformer):
    yaml_string = transformer.to_yaml()
    transformer_prime = utils.config.from_yaml(yaml_string)
    yaml_string_prime = transformer_prime.to_yaml()
    assert yaml_string == yaml_string_prime, "Expected {} to equal {}".format(yaml_string, yaml_string_prime)


@tools.params(Transformer(),
              OpsTransformer(scale=2.0, offset=-1.0),
              OpsTransformer(scale=2.0 / 255.0, offset=-1.0, exponent=-1, transpose=(2, 0, 1)),
              ImageTransformer(),
              CompositionTransformer([]),
              CompositionTransformer(
                  [OpsTransformer(scale=2.0 / 255.0, offset=-1.0, transpose=(2, 0, 1)),
                   ImageTransformer()]),
              )
def test_process_roundtripping(transformer):
    # TODO: ImageTransformer tests
    data = np.random.random((480, 640, 3))
    data_prime = transformer.deprocess(transformer.preprocess(data))
    assert np.allclose(data, data_prime), "Expected {} to equal {}".format(data, data_prime)


@tools.params(Transformer(),
              OpsTransformer(scale=2.0, offset=-1.0),
              OpsTransformer(scale=2.0 / 255.0, offset=-1.0, exponent=-1, transpose=(2, 0, 1)),
              ImageTransformer(),
              CompositionTransformer([]),
              CompositionTransformer(
                  [OpsTransformer(scale=2.0 / 255.0, offset=-1.0, transpose=(2, 0, 1)),
                   ImageTransformer()]),
              )
def test_process_shape_roundtripping(transformer):
    # TODO: ImageTransformer tests
    data = np.random.random((480, 640, 3))
    pre_data = transformer.preprocess(data)
    pre_shape_prime = transformer.preprocess_shape(data.shape)
    shape_prime = transformer.deprocess_shape(pre_data.shape)
    assert shape_prime == data.shape, "Expected {} to equal {}".format(shape_prime, data.shape)
    assert pre_shape_prime == pre_data.shape, "Expected {} to equal {}".format(pre_shape_prime, pre_data.shape)
