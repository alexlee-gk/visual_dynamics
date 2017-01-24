import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec


class LossPlotter:

    def __init__(self, fig, gs, format_strings=None, format_dicts=None, labels=None, xlabel=None, ylabel=None, yscale='linear'):
        self._fig = fig
        self._gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs)
        self._ax = plt.subplot(self._gs[0])

        self._labels = labels or []
        self._format_strings = format_strings or []
        self._format_dicts = format_dicts or []

        self._ax.set_xlabel(xlabel or 'iteration')
        self._ax.set_ylabel(ylabel or 'loss')
        self._ax.set_yscale(yscale or 'linear')
        self._ax.minorticks_on()

        self._plots = []

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()   # Fixes bug with Qt4Agg backend

    def update(self, all_losses, all_loss_iters=None):
        data_len = len(all_losses)
        self._plots += [None] * (data_len - len(self._plots))
        self._labels += [None] * (data_len - len(self._labels))
        self._format_strings += [''] * (data_len - len(self._format_strings))
        self._format_dicts += [dict()] * (data_len - len(self._format_dicts))

        all_loss_iters_ = []
        for i, (plot, label, format_string, format_dict, losses) in \
                enumerate(zip(self._plots, self._labels, self._format_strings, self._format_dicts, all_losses)):
            if all_loss_iters is not None and i < len(all_loss_iters) and all_loss_iters[i] is not None:
                loss_iters = all_loss_iters[i]
            else:
                loss_iters = np.arange(len(losses))
            all_loss_iters_.append(loss_iters)
            if plot is None:
                self._plots[i] = self._ax.plot(loss_iters, losses, format_string, label=label, **format_dict)[0]
            else:
                plot.set_data(loss_iters, losses)

        ylim = self._ax.get_ylim()
        if self._ax.get_yscale() == 'log':
            ylim = (min(1, ylim[0]), np.max(np.concatenate(all_losses)))
        else:
            ylim = (np.min(np.concatenate(all_losses)), np.max(np.concatenate(all_losses)))
        self._ax.set_ylim(ylim)
        xlim = (np.min(np.concatenate(all_loss_iters_)), np.max(np.concatenate(all_loss_iters_)))
        self._ax.set_xlim(xlim)
        self._ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
        self.draw(num_plots=data_len)

    def draw(self, num_plots=None):
        # self._ax.draw_artist(self._ax.patch)
        # for plot in self._plots[:num_plots]:
        #     self._ax.draw_artist(plot)
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()   # Fixes bug with Qt4Agg backend
