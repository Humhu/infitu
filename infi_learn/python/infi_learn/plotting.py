"""Various plotting tools
"""
import time
import math
import abc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import rospy
import numpy as np

import threading
from collections import deque

# TODO Make interface so MultiPlottable can inherit, move body to SinglePlottable base
class Plottable(object):
    """Interface for objects that need to be called from a main 
    GUI thread
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, other=None, xlabel='', ylabel='',
                 title=''):

        self.other = other
        if other is None:
            self.inited = threading.Event()
            self.fig = None
            self._ax = None
            self.xlabel = xlabel
            self.ylabel = ylabel
            self.title = title
        else:
            self.inited = other.inited
            self.fig = None
            self._ax = None

    @property
    def ax(self):
        if self.other is None:
            return self._ax
        else:
            self.other.wait_for_init()
            return self.other.ax

    def wait_for_init(self):
        self.inited.wait()

    def draw(self):
        """Calls the draw update method to put this objects
        plottable callbacks on the GUI queue
        """
        if self.fig is None:
            if self.other is None:
                self.fig = plt.figure()
                self._ax = plt.axes()
                self.inited.set()
                self._ax.set_xlabel(self.xlabel)
                self._ax.set_ylabel(self.ylabel)
                self._ax.set_title(self.title)
            else:
                self.fig = self.other.fig
                self._ax = self.other.ax

        self.fig.canvas.draw_idle()

class MultiPlottable(object):
    """Multi-axis wrapper class
    """
    def __init__(self, n_i, n_j):
        self.fig = None
        self.n_i = int(n_i)
        self.n_j = int(n_j)
        self.inited = threading.Event()
        self._axes = None

    def wait_for_init(self):
        self.inited.wait()
        
    def draw(self):
        """Calls the draw update method to put this objects
        plottable callbacks on the GUI queue
        """
        if self.fig is None:
            self.fig = plt.figure()
            self._axes = self.fig.subplots(nrows=self.n_j, ncols=self.n_i).flatten()
            self.inited.set()

        self.fig.canvas.draw_idle()

    def get_axis_handle(self, ind):
        return MultiPlottableHandle(self, ind)

class MultiPlottableHandle(object):
    def __init__(self, base, ind):
        self.base = base
        self.ind = ind

    @property
    def ax(self):
        self.base.wait_for_init()
        return self.base._axes[self.ind]

    @property
    def inited(self):
        return self.base.inited

    @property
    def fig(self):
        return self.base.fig

    def wait_for_init(self):
        self.base.wait_for_init()

class BlockingRequest(object):
    def __init__(self, func):
        self.func = func
        self.block = threading.Event()
        self.retval = None

    def process(self):
        self.retval = self.func()
        self.block.set()

    def wait(self):
        self.block.wait()
        return self.retval


class PlottingGroup(object):
    def __init__(self):
        self.plots = []
        self.requests = deque()

    def request_init(self, func):
        req = BlockingRequest(func)
        self.requests.append(req)
        return req.wait()

    def add_plottable(self, p):
        # TODO Add back in after Plottable is an interface
        # if not isinstance(p, Plottable):
            # raise ValueError('Must implement Plottable interface')
        self.plots.append(p)

    def draw_all(self):
        for p in self.plots:
            p.draw()
        if len(self.plots) > 0:
            plt.pause(0.01)

    def spin(self, rate):
        while not rospy.is_shutdown():
            self.draw_all()

            for req in self.requests:
                req.process()
            self.requests.clear()

            time.sleep(1.0 / rate)
        plt.close('all')


class LineSeriesPlotter(Plottable):
    def __init__(self, **kwargs):
        Plottable.__init__(self, **kwargs)
        self.objects = {}

    def _check_in(self, x):
        if np.iterable(x):
            return list(x)
        else:
            return [x]

    def add_line(self, name, x, y, **kwargs):
        self.wait_for_init()

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
        self.wait_for_init()

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
        self.wait_for_init()

        self.objects[name] = (self.ax.plot(x, y, label=name, **kwargs)[0],
                              x, y)
        self.ax.legend()

    def clear_line(self, name):
        self.wait_for_init()

        if name in self.objects:
            self.objects[name][0].remove()
            self.objects.pop(name)
            self.ax.legend()
            self.ax.relim()
            self.ax.autoscale_view(True, True, True)

class ScatterPlotter(Plottable):
    def __init__(self, cm=cm.bwr, **kwargs):
        Plottable.__init__(self, **kwargs)
        self.objects = {}
        self.cm = cm

    def add_scatter(self, name, x, y, c, **kwargs):
        self.wait_for_init()

        x = list(np.atleast_1d(x))
        y = list(np.atleast_1d(y))
        c = list(np.atleast_1d(c))

        if name not in self.objects:
            self._create_scatter(name, x, y, c, **kwargs)
        else:
            lh, xs, ys, cs = self.objects[name]
            xs.extend(x)
            ys.extend(y)
            cs.extend(c)
            self._create_scatter(name, xs, ys, cs, **kwargs)

    def set_scatter(self, name, x, y, c, **kwargs):
        self.wait_for_init()

        x = list(np.atleast_1d(x))
        y = list(np.atleast_1d(y))
        c = list(np.atleast_1d(c))
        self._create_scatter(name, x, y, c, **kwargs)

    def _create_scatter(self, name, x, y, c, **kwargs):
        self.wait_for_init()

        if name in self.objects:
            lh, xs, ys, cs = self.objects[name]
            lh.remove()
        self.objects[name] = (self.ax.scatter(x, y, c=c, cmap=self.cm, label=name, **kwargs),
                              x, y, c)

    def clear_scatter(self, name):
        self.wait_for_init()

        if name in self.objects:
            self.objects[name][0].remove()
            self.objects.pop(name)
            self.ax.relim()
            self.ax.autoscale_view(True, True, True)

class ImagePlotter(Plottable):
    def __init__(self, cm=cm.bwr, vmin=-1, vmax=1, noticks=True, **kwargs):
        Plottable.__init__(self, **kwargs)
        self.pim = None
        self.cm = cm
        self.vmin = vmin
        self.vmax = vmax
        self.noticks = noticks

    def set_image(self, img, extents=None):
        self.wait_for_init()

        if self.pim is None:
            if self.noticks:
                self.ax.axis('off')
            self.pim = self.ax.imshow(img,
                                      origin='lower',
                                      extent=extents,
                                      cmap=self.cm,
                                      vmin=self.vmin,
                                      vmax=self.vmax)
        else:
            self.pim.set_data(img)
            if extents is not None:
                self.pim.set_extent(extents)

class FilterPlotter(object):
    def __init__(self, N):
        s = math.ceil(math.sqrt(float(N)))
        self.base = MultiPlottable(s, s)
        self.plotters = []
        for i in range(N):
            p = ImagePlotter(other=self.base.get_axis_handle(i))
            self.plotters.append(p)

    def set_filters(self, filters):
        for i, p in enumerate(self.plotters):
            p.set_image(filters[i])

    def draw(self):
        self.base.draw()
        for p in self.plotters:
            p.draw()