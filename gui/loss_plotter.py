import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec


class LossPlotter:

    def __init__(self, fig, gs, labels=None):
        self._fig = fig
        self._gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs)
        self._ax = plt.subplot(self._gs[0])

        self._labels = labels or []

        self._ax.set_xlabel('iteration')
        self._ax.set_ylabel('loss')
        self._ax.minorticks_on()

        self._plots = []

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()   # Fixes bug with Qt4Agg backend

    def update(self, all_losses, all_loss_iters=None):
        data_len = len(all_losses)
        self._plots += [None] * (data_len - len(self._plots))
        self._labels += [None] * (data_len - len(self._labels))

        for i, (plot, label, losses) in enumerate(zip(self._plots, self._labels, all_losses)):
            if all_loss_iters is not None and i < len(all_loss_iters) and all_loss_iters[i] is not None:
                loss_iters = all_loss_iters[i]
            else:
                loss_iters = np.arange(len(losses))
            if plot is None:
                self._plots[i] = self._ax.plot(loss_iters, losses, label=label)[0]
            else:
                plot.set_data(loss_iters, losses)

        ylim = self._ax.get_ylim()
        ylim = (min(0, ylim[0]), min(2 * np.median(np.concatenate(all_losses)), ylim[1]))
        self._ax.set_ylim(ylim)
        self._ax.set_xlim((0, loss_iters[-1] if loss_iters[-1] > 0 else 1))
        self._ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
        self.draw(num_plots=data_len)

    def draw(self, num_plots=None):
        self._ax.draw_artist(self._ax.patch)
        for plot in self._plots[:num_plots]:
            self._ax.draw_artist(plot)
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()   # Fixes bug with Qt4Agg backend
