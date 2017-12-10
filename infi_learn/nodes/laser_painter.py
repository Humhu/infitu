#!/usr/bin/env python

import math
import numpy as np

import rospy
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge

class LaserImagePainter(object):
    """Converts planar laser scans to fixed-size 2D byte images.

    Parameters
    ----------
    dim  : int
        Dimensionality of the laser scan
    laser_fov  : 2-tuple or list of float
        Min and max angular sweep of scan in radians
    max_range  : float
        Max valid (or desired) range per beam
    resolution : float
        Image cell size in meters
    """

    def __init__(self, dim, fov, max_range, resolution,
                 dtype=np.uint8, empty_val=0, fill_val=1):
        angles = np.linspace(fov[0], fov[1], dim)
        self._cosses = np.cos(angles)
        self._sines = np.sin(angles)
        self.res = resolution
        self.max_range = max_range

        max_xs = self.max_range * self._cosses
        max_ys = self.max_range * self._sines
        self._offsets = (round(-min(max_xs) / self.res),
                         round(-min(max_ys) / self.res))
        # NOTE In our convention, the origin is in the middle of a square pixel
        self.img_size = (int(math.ceil((max(max_xs) - min(max_xs)) / self.res)) + 1,
                         int(math.ceil((max(max_ys) - min(max_ys)) / self.res)) + 1)
        self.empty_val = empty_val
        self.fill_val = fill_val
        self.dtype = dtype

    def scan_to_image(self, scan):
        """Converts planar laser scan to 2D image. Ranges less than 0,
        exceeding the max range, or non-finite are discarded.

        Parameters
        ----------
        scan : iterable of floats
            The scan ranges, assumed ordered from min angle to max angle

        Returns
        -------
        img  : numpy 2D array [width, height, 1]
            The painted image with 0s in empty space and 1s on hits
        """
        scan = np.array(scan)
        valid = np.logical_and(scan > 0.0, scan <= self.max_range)
        valid = np.logical_and(valid, np.isfinite(scan))
        x = scan * self._cosses
        y = scan * self._sines
        ix = np.round(x[valid] / self.res + self._offsets[0]).astype(int)
        iy = np.round(y[valid] / self.res + self._offsets[1]).astype(int)
        img = np.full(shape=(self.img_size[0], self.img_size[1], 1), dtype=self.dtype,
                      fill_value=self.empty_val)
        img[ix, iy] = self.fill_val
        return img

class LaserPainter(object):
    """Subscribes to a laser scan and paints them to a 2D image.
    """

    def __init__(self):

        self.painter = None
        self.paint_resolution = rospy.get_param('~paint_resolution', 0.1)
        self.max_range = rospy.get_param('~max_range')
        self.fov = rospy.get_param('~fov')

        self.bridge = CvBridge()
        self.laser_sub = rospy.Subscriber('scan', LaserScan,
                                          callback=self.scan_callback)
        self.image_pub = rospy.Publisher('image', Image, queue_size=10)

    def init_painter(self, dim):
        self.painter = LaserImagePainter(dim=dim,
                                         fov=self.fov,
                                         max_range=self.max_range,
                                         resolution=self.paint_resolution,
                                         dtype=np.uint8,
                                         empty_val=0,
                                         fill_val=255)
        rospy.loginfo('Enabled painting with image size: %s',
                      str(self.painter.img_size))

    def scan_callback(self, msg):
        if self.painter is None:
            self.init_painter(len(msg.ranges))

        image = self.painter.scan_to_image(np.array(msg.ranges))
        out_msg = self.bridge.cv2_to_imgmsg(image, "mono8")
        out_msg.header = msg.header
        self.image_pub.publish(out_msg)

if __name__ == '__main__':
    rospy.init_node('laser_painter')
    lp = LaserPainter()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
