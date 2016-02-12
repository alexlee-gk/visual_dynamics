import argparse
import numpy as np
import data_container


def rms_errors(current, target):
    assert current.ndim == target.ndim + 1
    num_trajs = len(target)
    rms_errors = np.sqrt(((target - current[:, -1, ...]).reshape((num_trajs, -1)) ** 2).mean(axis=1))
    assert rms_errors.ndim == 1
    return rms_errors

def mean_rms_error(current, target):
    return rms_errors(current, target).mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('traj_container_fnames', type=str, nargs='+')
    parser.add_argument('--traj_container', type=str, default='ImageTrajectoryDataContainer')
    args = parser.parse_args()

    TrajectoryDataContainer = getattr(data_container, args.traj_container)
    if not issubclass(TrajectoryDataContainer, data_container.TrajectoryDataContainer):
        raise ValueError('trajectory data container %s'%args.traj_data_container)
    for traj_container_fname in args.traj_container_fnames:
        print(traj_container_fname)
        try:
            traj_container = TrajectoryDataContainer(traj_container_fname)
            num_trajs = traj_container.num_trajs
            num_steps = traj_container.num_steps-1
            sim_args = traj_container.get_group('sim_args')
            vel_scale = sim_args['vel_scale']
            images_all = []
            dof_vals_all = []
            image_target_all = []
            dof_val_target_all = []
            car_dof_values_all = []
            for traj_iter in range(num_trajs):
                images = []
                dof_vals = []
                for step_iter in range(num_steps+1):
                    image, dof_val = traj_container.get_datum(traj_iter, step_iter, ['image_curr', 'dof_val']).values()
                    images.append(image)
                    dof_vals.append(dof_val)
                images = np.asarray(images)
                dof_vals = np.asarray(dof_vals)
                try:
                    image_target, dof_val_target, car_dof_values = traj_container.get_datum(traj_iter, num_steps, ['image_target', 'dof_values_target', 'car_dof_values']).values()
                    car_dof_values_all.append(car_dof_values)
                except:
                    image_target, dof_val_target = traj_container.get_datum(traj_iter, num_steps, ['image_target', 'dof_values_target']).values()
                images_all.append(images)
                dof_vals_all.append(dof_vals)
                image_target_all.append(image_target)
                dof_val_target_all.append(dof_val_target)
            images_all = np.asarray(images_all)
            dof_vals_all = np.asarray(dof_vals_all)
            image_target_all = np.asarray(image_target_all)
            dof_val_target_all = np.asarray(dof_val_target_all)
            car_dof_values_all = np.asarray(car_dof_values_all)
    
            mean_image_target_rms_error = mean_rms_error(images_all, image_target_all)
            mean_dof_val_target_rms_error = mean_rms_error(dof_vals_all / vel_scale, dof_val_target_all / vel_scale)
            if len(car_dof_values_all) > 0:
                mean_car_dof_val_target_rms_error = mean_rms_error(dof_vals_all[..., :3], car_dof_values_all)
            if len(car_dof_values_all) > 0:
                print('%.4f'%mean_image_target_rms_error, '%.2f'%mean_dof_val_target_rms_error, '%.2f'%mean_car_dof_val_target_rms_error)
            else:
                print('%.4f'%mean_image_target_rms_error, '%.2f'%mean_dof_val_target_rms_error)
        except:
            pass

if __name__ == "__main__":
    main()
