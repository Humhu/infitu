#!/usr/bin/env python

import rospy

from geometry_msgs.msg import Twist
from infi_msgs.msg import EpisodeBreak


class VelocityMonitor(object):
    """Freezes velocity commands upon episode break
    """

    def __init__(self):

        self.cmd_sub = rospy.Subscriber('cmd_vel', Twist, self.cmd_callback,
                                        queue_size=10)
        self.cmd_pub = rospy.Publisher('out_vel', Twist, queue_size=10)
        self.break_sub = rospy.Subscriber('break', EpisodeBreak, self.break_callback,
                                          queue_size=10)
        # NOTE In theory don't need mutex since rospy messages are single
        # threaded
        self.is_enabled = True

    def cmd_callback(self, msg):
        if not self.is_enabled:
            # NOTE Initializes to zeros
            msg = Twist()
            rospy.loginfo('Outside of active episode, commanding zero reference!')
        self.cmd_pub.publish(Twist())

    def break_callback(self, msg):
        self.is_enabled = msg.start
        if msg.start:
            rospy.loginfo('Episode active, passing references')
        else:
            rospy.loginfo('Episode inactive, zeroing references')

if __name__ == '__main__':
    rospy.init_node('velocity_monitor')
    vm = VelocityMonitor()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
