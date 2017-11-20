#! /usr/bin/env python

import infitu as rr
import rospy

if __name__ == '__main__':
    rospy.init_node('test_laser')

    backend = rr.MonolithicBackend()
    fe = rr.LaserFrontend(backend)
    try:
        while not rospy.is_shutdown():
            fe.spin(rospy.Time.now().to_sec())
    except rospy.ROSInterruptException:
        pass
    