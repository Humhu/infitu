"""Various plotting tools
"""
import time
import abc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import rospy
import numpy as np


class Plottable(object):
    """Interface for objects that need to be called from a main 
    GUI thread
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, ax=None):
        if ax is None:
            self.fig = plt.figure()
            self.ax = plt.axes()
        else:
            self.ax = ax
            self.fig = ax.figure

    def draw(self):
        """Calls the draw update method to put this objects
        plottable callbacks on the GUI queue
        """
        self.fig.canvas.draw_idle()


class PlottingGroup(object):
    def __init__(self):
        # plt.ion()
        self.plots = []

    def add_plottable(self, p):
        if not isinstance(p, Plottable):
            raise ValueError('Must implement Plottable interface')
        self.plots.append(p)

    def draw_all(self):
        for p in self.plots:
            p.draw()
        plt.pause(0.01)

    def spin(self, rate):
        while not rospy.is_shutdown():
            self.draw_all()
            time.sleep(1.0 / rate)
        plt.close('all')


class LineSeriesPlotter(Plottable):
    def __init__(self, xlabel='', ylabel='', title='',
                 ax=None):
        Plottable.__init__(ax)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        self.objects = {}

    def _check_in(self, x):
        if np.iterable(x):
            return list(x)
        else:
            return [x]

    def add_line(self, name, x, y, **kwargs):
        # self._focus()
        x = list(np.atleast_1d(x))
        y = list(np.atleast_1d(y))

        if name not in self.objects:
            self._create_line(name, x, y, **kwargs)
        else:
            lh, xs, ys = self.objects[name]
            xs.extend(x)
            ys.extend(y)
            lh.set_data(xs, ys)
            self.ax.relim()
            self.ax.autoscale_view(True, True, True)

    def set_line(self, name, x, y, **kwargs):
        # self._focus()
        x = list(np.atleast_1d(x))
        y = list(np.atleast_1d(y))

        if name not in self.objects:
            self._create_line(name, x, y, **kwargs)
        else:
            lh = self.objects[name][0]
            self.objects[name] = (lh, x, y)
            lh.set_data(x, y)
            self.ax.relim()
            self.ax.autoscale_view(True, True, True)

    def _create_line(self, name, x, y, **kwargs):
        self.objects[name] = (self.ax.plot(x, y, label=name, **kwargs)[0],
                              x, y)
        self.ax.legend()

    def clear_line(self, name):
        if name in self.objects:
            self.objects[name][0].remove()
            self.objects.pop(name)
            self.ax.legend()
            self.ax.relim()
            self.ax.autoscale_view(True, True, True)


class ImagePlotter(Plottable):
    def __init__(self, xlabel='', ylabel='', title='',
                 cm=cm.bwr, vmin=-1, vmax=1, ax=None):
        Plottable.__init__(ax)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        self.pim = None
        self.cm = cm
        self.vmin = vmin
        self.vmax = vmax

    def set_image(self, img, extents=None):
        if self.pim is None:
            self.pim = self.ax.imshow(img,
                                      origin='lower',
                                      extent=extents,
                                      cmap=self.cm,
                                      vmin=self.vmin,
                                      vmax=self.vmax)
        else:
            self.pim.set_data(img)
