#!/usr/bin/env python

import numpy as np

from infi_learn import LaserImagePainter
import rospy
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge, CvBridgeError

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
        self.image_pub = rospy.Publisher('image', Image)

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
