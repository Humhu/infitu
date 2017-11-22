#!/usr/bin/env python

import rospy
import numpy as np
import random
from percepto_msgs.srv import SetParameters, SetParametersRequest
from percepto_msgs.msg import EpisodeBreak


class RandomActions(object):
    """Performs random actions at a fixed rate.
    """

    def __init__(self):
        self.dim = rospy.get_param('~dim')

        def farr(x):
            if hasattr(x, '__iter__'):
                return np.array([float(xi) for xi in x])
            else:
                return np.full(self.dim, float(x))
        self.scale = farr(rospy.get_param('~scale', 1.0))
        self.offset = farr(rospy.get_param('~offset', 0.0))

        self.sample_mode = rospy.get_param('~sample_mode', 'uniform')
        if self.sample_mode == 'discrete':
            n = rospy.get_param('~num_actions')
            self.actions = [self._sample_action() for _ in range(n)]

        set_topic = rospy.get_param('~set_topic')
        self.param_proxy = rospy.ServiceProxy(set_topic, SetParameters,
                                              persistent=True)

        trigger_mode = rospy.get_param('~trigger_mode', 'timer')
        if trigger_mode == 'timer':
            rate = rospy.get_param('~timer_rate')
            self.timer = rospy.Timer(rospy.Duration(1.0 / rate),
                                     callback=self.timer_callback)
        elif trigger_mode == 'break':
            self.break_sub = rospy.Subscriber('break', EpisodeBreak,
                                              self.break_callback)
        else:
            raise ValueError('Unknown trigger mode: %s' % trigger_mode)

        self._set_action()

    def _sample_action(self):
        return np.random.uniform(low=-1, high=1, size=self.dim) * self.scale + self.offset

    def _set_action(self):
        req = SetParametersRequest()
        if self.sample_mode == 'discrete':
            req.parameters = random.choice(self.actions)
        elif self.sample_mode == 'uniform':
            req.parameters = self._sample_action()
        else:
            raise ValueError('Unknown sampling mode: %s' % self.sample_mode)

        try:
            self.param_proxy.call(req)
        except rospy.ServiceException:
            rospy.logerr('Could not set parameters to: %s' %
                         str(req.parameters))

    def break_callback(self, event):
        if event.start:
            return
        rospy.loginfo('Break received with cause: %s at %f, setting action',
                      event.cause, event.time.to_sec())
        self._set_action()

    def timer_callback(self, event):
        rospy.loginfo('Timer triggered at %f, setting action' %
                      event.current_real.to_sec())
        self._set_action()


if __name__ == '__main__':
    rospy.init_node('random_actions')
    ra = RandomActions()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
