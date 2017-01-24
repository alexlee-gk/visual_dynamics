from __future__ import division, print_function

import cv2
import matplotlib.pyplot as plt
import numpy as np


def visualize_images_callback(images, image_transformer=None, depth_scale=None, vis_scale=10,
                              window_name='Image window', delay=1):
    if image_transformer:
        images = [image_transformer.deprocess(image) for image in images]
    vis_images = []
    for image in images:
        if image.dtype != np.uint8:
            if depth_scale is None:
                depth_scale = image.max()
            image = np.clip(image / depth_scale, 0.0, 1.0)
            image = (255.0 * image).astype(np.uint8)
        if image.ndim == 2:
            image = image[..., None]
        if image.shape[2] == 1:
            image = np.tile(image, (1, 1, 3))
        vis_images.append(image)
    vis_image = np.concatenate(vis_images, axis=1)
    if vis_scale != 1:
        vis_image = cv2.resize(vis_image, (0, 0), fx=vis_scale, fy=vis_scale, interpolation=cv2.INTER_NEAREST)
    else:
        vis_image = vis_image
    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, vis_image)
    key = cv2.waitKey(delay)
    key &= 255
    if key == 27 or key == ord('q'):
        print("Pressed ESC or q, exiting")
        exit_request = True
    else:
        exit_request = False
    return exit_request, key


def draw_images_callback(images, image_transformer=None, window_name=None, num=None, figsize=(16, 12)):
    # DEPRECATED: use grid_image_visualizer instead
    plt.ion()
    if isinstance(images, np.ndarray):
        # promote array to list of arrays
        images = [images]
    if isinstance(images[0], np.ndarray):
        # promote list of arrays to list of lists of arrays
        images = [images]
    nrows = len(images)
    ncols = 1  # at least
    for images_list in images:
        if not isinstance(images_list, (list, tuple)):
            raise ValueError("images are not formatted properly")
        ncols = max(ncols, len(images_list))
        for image_array in images_list:
            if not isinstance(image_array, np.ndarray):
                raise ValueError("images are not formatted properly")
    for images_list in images:
        if len(images_list) != ncols:
            images_list.extend([None] * (ncols - len(images_list)))
    fig = plt.figure(num=num, figsize=figsize, frameon=False, tight_layout=True)
    if len(fig.axes) == nrows * ncols:
        # assume figure is setup correctly
        axarr = np.array(fig.axes, dtype=object).reshape(nrows, ncols)
    else:
        fig.clear()
        fig, axarr = plt.subplots(nrows, ncols, num=num, figsize=figsize, squeeze=False, frameon=False, tight_layout=True)
    if window_name is not None:
        fig.canvas.set_window_title(window_name)
    for irow in range(nrows):
        for icol in range(ncols):
            ax = axarr[irow, icol]
            image = images[irow][icol]
            if image is None:
                ax.clear()
                continue
            if image.ndim == 3 and image.shape[0] == 3:
                image = image_transformer.deprocess(image)
            if len(ax.images) == 1:
                im = ax.images[0]
                im.set_data(image)
            else:
                ax.clear()
                ax.imshow(image, interpolation='none')
                ax.tick_params(axis='both', which='both', length=0, labelleft='off', labelbottom='off')
    plt.draw()
    return fig, axarr


def vis_square(data, grid_shape=None, padsize=1, padval=0, data_min=None, data_max=None):
    """
    take an array of shape (n, height, width) or (n, height, width, channels)
    and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
    """
    data_min = data_min if data_min is not None else data.min()
    data_max = data_max if data_max is not None else data.max()
    data = (data - data_min) / (data_max - data_min)

    num_filters = data.shape[0]
    if grid_shape is None:
        # force the number of filters to be square
        nrows = ncols = int(np.ceil(np.sqrt(num_filters)))
    else:
        nrows, ncols = grid_shape
    assert num_filters <= nrows * ncols

    padding = ((0, nrows * ncols - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((nrows, ncols) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((nrows * data.shape[1], ncols * data.shape[3]) + data.shape[4:])
    return data
