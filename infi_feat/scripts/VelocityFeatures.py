#!/usr/bin/env python

import rospy
import broadcast

from geometry_msgs.msg import TwistStamped


class VelocityFeatures(object):
    def __init__(self):
        self.mode = rospy.get_param('dim_mode')
        if self.mode == '2d':
            dim = 3
        elif self.mode == '3d':
            dim = 6
        else:
            raise ValueError('Unknown mode %s' % self.mode)

        topic = rospy.get_param('~stream_name')
        self.tx = broadcast.Transmitter(stream_name=topic,
                                        feature_size=dim,
                                        description='Velocity components',
                                        mode='push')

        self.velsub = rospy.Subscriber('twist',
                                       TwistStamped,
                                       self.twist_callback)

    def twist_callback(self, msg):
        if self.mode == '2d':
            feats = [msg.linear.x, msg.linear.y, msg.angular.z]
        elif self.mode == '3d':
            feats = [msg.linear.x, msg.linear.y, msg.linear.z,
                     msg.angular.x, msg.angular.y, msg.angular.z]
        self.tx.publish(time=msg.header.stamp,
                        feats=feats)


if __name__ == '__main__':
    rospy.init_node('velocity_features')
    vf = VelocityFeatures()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
