#!/usr/bin/env python

import rospy
import cv2 as cv
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import broadcast


class FloorFeaturesNode:
    """Generates sharpness features from a stream of images.
    """
    def __init__(self):

        self.scale = rospy.get_param('~scale')

        self.lap_k = rospy.get_param('~laplacian_k', 1)
        self.bridge = CvBridge()

        self.num_pyrs = rospy.get_param('~pyramid_levels', 0)

        stream_name = rospy.get_param('~stream_name')
        self.feat_tx = broadcast.Transmitter(stream_name=stream_name,
                                             namespace='~',
                                             feature_size=self.num_pyrs + 1,
                                             description=['laplacian_std'],
                                             mode='push',
                                             queue_size=0)

        buff_size = rospy.get_param('~buffer_size', 1)
        self.image_sub = rospy.Subscriber('image', Image,
                                          self.ImageCallback,
                                          queue_size=buff_size)

    def ImageCallback(self, msg):

        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")

        features = []
        for i in range(0, self.num_pyrs + 1):
            lap = cv.Laplacian(img, cv.CV_32F, None, self.lap_k)
            lap_std = np.std(lap.flat) * self.scale
            features.append(lap_std)
            img = cv.pyrDown(img)
        self.feat_tx.publish(msg.header.stamp, features)


if __name__ == '__main__':
    rospy.init_node('floor_features_node')
    try:
        ffn = FloorFeaturesNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
