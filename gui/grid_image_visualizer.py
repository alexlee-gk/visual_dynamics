import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import utils


matplotlib.rcParams['image.cmap'] = 'viridis'


class GridImageVisualizer(object):

    def __init__(self, fig, gs, num_plots=None, rows=None, cols=None, labels=None, vs_grid_shape=None, vs_padsize=1):
        """
        Args:
            vs_grid_shape: used for vis_square
            vs_padsize: used for vis_square
        """
        if num_plots is None:
            assert not (rows is None and cols is None)
            num_plots = rows * cols
        if cols is None:
            if rows is None:
                cols = int(np.floor(np.sqrt(num_plots)))
            else:
                cols = int(np.ceil(float(num_plots) / rows))
        if rows is None:
            rows = int(np.ceil(float(num_plots) / cols))
        assert num_plots <= rows * cols, 'Too many plots to put into gridspec.'
        self._rows = rows
        self._cols = cols

        self._fig = fig
        self._gs_image_axis = gs
        self._vs_grid_shape = vs_grid_shape
        self._vs_padsize = vs_padsize

        self._gs_image_axes = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=self._gs_image_axis)
        self._axarr_image = [plt.subplot(self._gs_image_axes[i]) for i in range(num_plots)]
        self._plots = [None] * num_plots

        for ax_image in self._axarr_image:
            ax_image.tick_params(axis='both', which='both', length=0, labelleft='off', labelbottom='off')
        labels = labels or []
        for ax_image, label in zip(self._axarr_image, labels):
            ax_image.set_xlabel(label)

        self._images = None

        def onpick(event):
            if event.artist in self._plots:
                need_redraw = False
                for plot, image in zip(self._plots, self._images):
                    if plot == event.artist:
                        if image.ndim == 3 and image.shape[2] != 3 and image.shape[0] > 3:
                            vis_image = plot.get_array()
                            if vis_image.shape == image.shape[1:]:
                                vis_image = utils.vis_square(image, grid_shape=self._vs_grid_shape, padsize=self._vs_padsize)
                            else:
                                x, y = event.mouseevent.inaxes.transData.inverted().transform((event.mouseevent.x, event.mouseevent.y))
                                h, w = image.shape[1:]
                                irow = int(y + 0.5) // (h + self._vs_padsize)
                                icol = int(x + 0.5) // (w + self._vs_padsize)
                                nrows = vis_image.shape[0] // (h + self._vs_padsize)
                                ncols = vis_image.shape[1] // (w + self._vs_padsize)
                                assert irow < nrows
                                assert icol < ncols
                                channel_ind = irow * ncols + icol
                                if channel_ind >= image.shape[0]:
                                    continue
                                vis_image = image[channel_ind]
                            plot.set_data(vis_image)
                            need_redraw = True
                if need_redraw:
                    self.draw()

        fig.canvas.mpl_connect('pick_event', onpick)

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()   # Fixes bug with Qt4Agg backend

    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols

    def update(self, images):
        self._images = images
        if images is None:
            return

        image_min = None
        image_max = None
        for image in images:
            if image.ndim == 3 and image.shape[2] != 3:
                if image_min is None or image.min() < image_min:
                    image_min = image.min()
                if image_max is None or image.max() > image_max:
                    image_max = image.max()
        for i, (ax_image, plot, image) in enumerate(zip(self._axarr_image, self._plots, images)):
            if image is None:
                if ax_image in self._fig.axes:
                    self._fig.delaxes(ax_image)
            else:
                if ax_image not in self._fig.axes:
                    self._fig.add_axes(ax_image)
                if image.ndim == 3 and image.shape[2] != 3:
                    if image.shape[0] == 3:
                        image = (image - image_min) / (image_max - image_min)
                        image = image.transpose((1, 2, 0))
                    elif image.shape[0] > 1:
                        image = utils.vis_square(image, grid_shape=self._vs_grid_shape, padsize=self._vs_padsize,
                                                 data_min=image_min, data_max=image_max)
                    elif image.shape[0] == 1:
                        image = np.squeeze(image, axis=0)
                    elif image.shape[-1] == 1:
                        image = np.squeeze(image, axis=-1)
                if plot is None or image.shape != plot.get_array().shape:
                    ax_image.images[:] = []
                    self._plots[i] = ax_image.imshow(image, interpolation='none', aspect='equal', picker=True)
                else:
                    plot.set_data(image)
        self.draw(num_plots=len(images))  # update the minimum number of necessary axes

    def draw(self, num_plots=None):
        # for ax_image, plot in zip(self._axarr_image[:num_plots], self._plots[:num_plots]):
        #     if ax_image not in self._fig.axes or plot is None:
        #         continue
        #     ax_image.draw_artist(ax_image.patch)
        #     ax_image.draw_artist(plot)
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()   # Fixes bug with Qt4Agg backend
