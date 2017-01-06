

def flatten_tree(tree, base_type=str):
    if isinstance(tree, base_type):
        return [tree]
    else:
        return sum([flatten_tree(child, base_type=base_type) for child in tree], [])


def unflatten_tree(tree, flat_list, base_type=str, copy_list=True):
    if copy_list:
        flat_list = list(flat_list)
    if isinstance(tree, base_type):
        return flat_list.pop(0)
    else:
        return [unflatten_tree(child, flat_list, base_type=base_type, copy_list=False) for child in tree]
