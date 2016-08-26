import numpy as np
from nose2 import tools
import tempfile
from utils import DataContainer, DataGenerator


container_fnames = [tempfile.mktemp() for _ in range(4)]
num_steps_per_traj = [100] * 4 + [50] * 12 + [100] + [150] * 6
num_trajs_per_container = [4, 12, 1, 6]
num_steps_per_container = [400, 600, 100, 900]
assert sum(num_steps_per_traj) == sum(num_steps_per_container)

for container_ind, container_fname in enumerate(container_fnames):
    with DataContainer(container_fname, 'x') as container:
        num_trajs = num_trajs_per_container[container_ind]
        num_steps = num_steps_per_container[container_ind] // num_trajs
        assert num_steps_per_container[container_ind] == num_steps * num_trajs
        container.reserve(['container_ind', 'traj_iter', 'step_iter'], (num_trajs, num_steps))
        for traj_iter in range(num_trajs):
            for step_iter in range(num_steps):
                container.add_datum(traj_iter, step_iter,
                                    container_ind=np.array(container_ind),
                                    traj_iter=np.array(traj_iter),
                                    step_iter=np.array(step_iter))


@tools.params([(0, 1), (0, 1)],
              [(0, 2), (0, 2)],
              [(0, 3), (0, 3)],
              [(0, 1), (0, 3)],
              [(0, 3), (0, 1)],
              [(-1, 2), (0, 3)],
              [(-2, 2), (0, 3)],
              [(-3, 2), (0, 3)]
              )
def test_generator_int_offset(offset_limits):
    traj_offset_limit, step_offset_limit = offset_limits
    data_name_offset_pairs = [('container_ind', 0),
                              *[('traj_iter', i) for i in range(*traj_offset_limit)],
                              *[('step_iter', i) for i in range(*step_offset_limit)]]
    generator = DataGenerator(*container_fnames,
                              data_name_offset_pairs=data_name_offset_pairs,
                              batch_size=32,
                              shuffle=True,
                              once=True)

    max_iter = 4
    for _iter, batch_data in zip(range(max_iter), generator):
        traj_iters_traj = np.array(batch_data[1:1 + (traj_offset_limit[1] - traj_offset_limit[0])])
        step_iters_traj = np.array(batch_data[-(step_offset_limit[1] - step_offset_limit[0]):])

        # all traj_iters should be the same
        assert (traj_iters_traj == traj_iters_traj[0, :]).all()
        # all consecutive step_iters should differ by 1
        assert ((step_iters_traj - np.arange(len(step_iters_traj))[:, None]) == step_iters_traj[0, :]).all()



@tools.params([(0, 1), (0, 1)],
              [(0, 2), (0, 2)],
              [(0, 3), (0, 3)],
              [(0, 1), (0, 3)],
              [(0, 3), (0, 1)],
              [(-1, 2), (0, 3)],
              [(-2, 2), (0, 3)],
              [(-3, 2), (0, 3)]
              )
def test_generator_slice_offset(offset_limits):
    traj_offset_limit, step_offset_limit = offset_limits
    data_name_offset_pairs = [('container_ind', 0),
                              ('traj_iter', slice(*traj_offset_limit)),
                              ('step_iter', slice(*step_offset_limit))]
    generator = DataGenerator(*container_fnames,
                              data_name_offset_pairs=data_name_offset_pairs,
                              batch_size=32,
                              shuffle=True,
                              once=True)

    max_iter = 4
    for _iter, batch_data in zip(range(max_iter), generator):
        traj_iters_traj = np.swapaxes(batch_data[1], 0, 1)
        step_iters_traj = np.swapaxes(batch_data[2], 0, 1)

        # all traj_iters should be the same
        assert (traj_iters_traj == traj_iters_traj[0, :]).all()
        # all consecutive step_iters should differ by 1
        assert ((step_iters_traj - np.arange(len(step_iters_traj))[:, None]) == step_iters_traj[0, :]).all()


@tools.params([(0, 1), (0, 1)],
              [(0, 2), (0, 2)],
              [(0, 3), (0, 3)],
              [(0, 1), (0, 3)],
              [(0, 3), (0, 1)],
              [(-1, 2), (0, 3)],
              [(-2, 2), (0, 3)],
              [(-3, 2), (0, 3)]
              )
def test_generator_list_offset(offset_limits):
    traj_offset_limit, step_offset_limit = offset_limits
    data_name_offset_pairs = [('container_ind', 0),
                              ('traj_iter', list(range(*traj_offset_limit))),
                              ('step_iter', list(range(*step_offset_limit)))]
    generator = DataGenerator(*container_fnames,
                              data_name_offset_pairs=data_name_offset_pairs,
                              batch_size=32,
                              shuffle=True,
                              once=True)

    max_iter = 4
    for _iter, batch_data in zip(range(max_iter), generator):
        traj_iters_traj = np.swapaxes(batch_data[1], 0, 1)
        step_iters_traj = np.swapaxes(batch_data[2], 0, 1)

        # all traj_iters should be the same
        assert (traj_iters_traj == traj_iters_traj[0, :]).all()
        # all consecutive step_iters should differ by 1
        assert ((step_iters_traj - np.arange(len(step_iters_traj))[:, None]) == step_iters_traj[0, :]).all()
