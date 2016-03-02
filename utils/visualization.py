import cv2
import numpy as np


def visualize_images_callback(*images, vis_scale=10, window_name='Image window', delay=1, ret_key=False):
    vis_image = np.concatenate([image for image in images], axis=1)
    if vis_scale != 1:
        vis_rescaled_image = cv2.resize(vis_image, (0, 0), fx=vis_scale, fy=vis_scale, interpolation=cv2.INTER_NEAREST)
    else:
        vis_rescaled_image = vis_image
    cv2.imshow(window_name, vis_rescaled_image)
    key = cv2.waitKey(delay)
    key &= 255
    if key == 27 or key == ord('q'):
        print("Pressed ESC or q, exiting")
        exit_request = True
    else:
        exit_request = False
    if ret_key:
        return vis_image, exit_request, key
    else:
        return vis_image, exit_request


def vis_square(data, padsize=1, padval=0, data_min=None, data_max=None):
    """
    take an array of shape (n, height, width) or (n, height, width, channels)
    and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
    """
    data_min = data_min or data.min()
    data_max = data_max or data.max()
    data = (data - data_min) / (data_max - data_min)

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    return data
