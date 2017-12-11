"""Data subscription and conversion sources to be used with frontends
"""

import abc
from collections import deque
import numpy as np

import rospy
from sensor_msgs.msg import Image, LaserScan

from cv_bridge import CvBridge, CvBridgeError
from broadcast.msg import FloatVectorStamped
from argus_utils import TimeSeries

import matplotlib.pyplot as plt


class DataSource(object):
    """Base interface for all timestamped data sources
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        # Fill with (time,data) tuples in order
        self.buffer = deque()

    def buffer_data(self, t, data):
        self.buffer.append((t, data))

    def get_data(self, until):
        """Returns a list of (time, data) tuples with times before
        the specified time.
        """
        data = []
        while len(self.buffer) > 0 and self.buffer[0][0] < until:
            data.append(self.buffer.popleft())
        return data


class MultiDataSource(object):
    """Synchronizers multiple data sources. Uses the first
    source as the primary source.
    """

    def __init__(self, sources, tol):
        super(MultiDataSource, self).__init__()
        self.sources = sources
        self.buffers = [TimeSeries() for _ in range(len(self.sources) - 1)]
        self.tol = tol

    def get_data(self, until):
        # First get all data to sync to
        primaries = self.sources[0].get_data(until)
        for i, src in enumerate(self.sources[1:]):
            secondaries = src.get_data(until)
            for t, d in secondaries:
                self.buffers[i].insert(t, d)
    

        out = []
        for t, data in primaries:
            closest = [self.buffers[i].get_closest_either(t)
                       for i in range(len(self.sources) - 1)]
            if any([i is None or (abs(t - i.time) > self.tol) 
                    for i in closest]):
                continue

            agg = [data] + [i.data for i in closest]
            out.append((t,agg))

        for b in self.buffers:
            b.trim_earliest_to(until - self.tol)
        return out

class VectorSource(DataSource):
    """Subscribes to a stamped vector
    """

    def __init__(self, topic):
        super(VectorSource, self).__init__()
        self.dim = None
        self.vec_sub = rospy.Subscriber(topic, FloatVectorStamped,
                                        callback=self._vec_callback)

    def _vec_callback(self, msg):
        if self.dim is None:
            self.dim = len(msg.values)
        self.buffer_data(t=msg.header.stamp.to_sec(),
                         data=msg.values)


class LaserSource(DataSource):
    """Subscribes to a laser scan
    """

    def __init__(self, topic, max_range=float('inf'), nan_value=-1,
                 inf_value=-1):
        super(LaserSource, self).__init__()

        self.dim = None
        self.laser_nan_val = nan_value
        self.laser_inf_val = inf_value

        self.laser_sub = rospy.Subscriber(topic, LaserScan,
                                          callback=self._scan_callback)

    def _scan_callback(self, msg):
        if self.dim is None:
            self.dim = len(msg.ranges)
        self.buffer_data(t=msg.header.stamp.to_sec(),
                         data=self._proc_scan(msg))

    def _proc_scan(self, scan):
        ranges = np.array(scan.ranges)
        ranges[np.isnan(ranges)] = self.laser_nan_val
        ranges[ranges == float('inf')] = self.laser_inf_val
        return ranges

class ImageSource(DataSource):
    """Subscribes to an image topic.
    """

    def __init__(self, topic, image_mode='gray'):
        super(ImageSource, self).__init__()
        self.dim = None
        self.image_mode = image_mode
        self.num_image_channels()  # NOTE Arg checking
        self.image_bridge = CvBridge()

        self.camera_sub = rospy.Subscriber(topic, Image,
                                           callback=self._image_callback)

    def num_image_channels(self):
        if self.image_mode == 'gray':
            return 1
        elif self.image_mode == 'rgb':
            return 3
        else:
            raise ValueError('Invalid image mode: %s' % self.image_mode)

    def _image_callback(self, msg):

        if self.image_mode == 'gray':
            try:
                image = self.image_bridge.imgmsg_to_cv2(msg, "mono8")
            except CvBridgeError as e:
                rospy.logerr('Could not convert image: %s', str(e))
                return
        elif self.image_mode == 'rbg':
            try:
                image = self.image_bridge.imgmsg_to_cv2(msg, "rgb8")
            except CvBridgeError as e:
                rospy.logerr('Could not convert image: %s', str(e))
                return

        if self.dim is None:
            self.dim = image.shape
        image = image.astype(float) / 255.0
        self.buffer_data(t=msg.header.stamp.to_sec(),
                         data=image)
