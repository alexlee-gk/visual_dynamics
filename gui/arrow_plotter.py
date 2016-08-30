import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec


class ArrowPlotter:

    def __init__(self, fig, gs, labels=None, limits=None):
        self._fig = fig
        self._gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs)
        self._ax = plt.subplot(self._gs[0])
        self._arrow = None
        if labels:
            if len(labels) == 2:
                self._ax.set_xlabel(labels[0])
                self._ax.set_ylabel(labels[1])
            else:
                raise ValueError("invalid labels %r" % labels)
        if limits:
            if len(limits) == 2 and \
                    len(limits[0]) == 2 and \
                    len(limits[1]) == 2:
                self._ax.set_xlim([limits[0][0], limits[1][0]])
                self._ax.set_ylim([limits[0][1], limits[1][1]])
            else:
                raise ValueError("invalid limits %r" % limits)
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()   # Fixes bug with Qt4Agg backend

    def update(self, value):
        if len(value) != 2:
            raise ValueError("invalid value %r" % value)
        self._ax.artists[:] = []
        self._arrow = self._ax.arrow(0.0, 0.0, *value,
                                     **dict(head_width=0.05, head_length=0.1, fc='k', ec='k'))
        self.draw()

    def draw(self):
        self._ax.draw_artist(self._ax.patch)
        if self._arrow is not None:
            self._ax.draw_artist(self._arrow)
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()   # Fixes bug with Qt4Agg backend
