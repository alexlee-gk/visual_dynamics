from nose2 import tools

from visual_dynamics import utils


@tools.params('0',
              ['0', '1'],
              ['0', ['1', '2']],
              [['0', '1'], '2'],
              [['0', '1', '2'], ['3', '4', '5']])
def test_flatten_unflatten_tree(tree):
    flat_list = utils.flatten_tree(tree)
    recon_tree = utils.unflatten_tree(tree, flat_list)
    print(tree, flat_list, recon_tree)
    assert tree == recon_tree
