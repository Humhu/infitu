#!/usr/bin/env python

import rospy

from infi_msgs.msg import RewardStamped, EpisodeBreak
from fieldtrack.srv import ResetFilter, ResetFilterRequest
from argus_utils import wait_for_service


class EpisodeTerminator(object):
    """Monitors a reward signal and publishes an episode termination message
    when the reward drops past a threshold. Also resets the filter
    """

    def __init__(self):

        reset_topic = rospy.get_param('~reset_service')
        wait_for_service(reset_topic)
        self.reset_proxy = rospy.ServiceProxy(reset_topic,
                                              ResetFilter)

        self.min_reward = rospy.get_param('~min_reward')

        self.break_pub = rospy.Publisher('~breaks', EpisodeBreak, queue_size=1)
        self.reward_sub = rospy.Subscriber('reward', RewardStamped,
                                           queue_size=1,
                                           callback=self.reward_callback)
        
        self.start_delay = rospy.get_param('~start_delay')
        self.episode_start = rospy.Time.now()

    def reward_callback(self, msg):
        if msg.reward > self.min_reward or msg.header.stamp < self.episode_start:
            return
        
        out = EpisodeBreak()
        out.time = msg.header.stamp
        out.start = False
        out.cause = 'Reward %f less than threshold %f' % (msg.reward, self.min_reward)
        self.break_pub.publish(out)
        rospy.loginfo(out.cause)

        req = ResetFilterRequest()
        req.time_to_wait = 0
        req.filter_time.set(0,0)
        try:
            self.reset_proxy.call(req)
        except rospy.ServiceException:
            rospy.logerr('Could not reset filter!')

        rospy.sleep(self.start_delay)
        out = EpisodeBreak()
        out.time = rospy.Time.now()
        out.start = True
        out.cause = 'Starting new episode after %f second pause' % self.start_delay
        self.break_pub.publish(out)
        rospy.loginfo(out.cause)
        self.episode_start = out.time

if __name__ == '__main__':
    rospy.init_node('episode_terminator')
    et = EpisodeTerminator()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
