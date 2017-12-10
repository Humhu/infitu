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
        self.wait_for_reward = rospy.get_param('~wait_for_reward', True)

        self.active = True

        self.reward_sub = rospy.Subscriber('reward', RewardStamped,
                                           queue_size=1,
                                           callback=self.reward_callback)

    def reward_callback(self, msg):

        if not self.active and msg.reward > self.min_reward:
            self.active = True

            # Publish the start of the new episode
            out = EpisodeBreak()
            out.start = True
            out.cause = 'Starting episode: reward %f at time %s greater than threshold %f' \
                % (msg.reward, str(msg.header.stamp), self.min_reward)
            out.time = rospy.Time.now()
            rospy.loginfo(out.cause)
            self.break_pub.publish(out)
            return

        elif self.active and msg.reward < self.min_reward:
            self.active = False

            # Publish the end of the episode
            out = EpisodeBreak()
            out.time = msg.header.stamp
            out.start = False
            out.cause = 'Reward %f at time %s less than threshold %f' \
                % (msg.reward, str(msg.header.stamp), self.min_reward)
            rospy.loginfo(out.cause)
            self.break_pub.publish(out)
            return

        # # Reset the filter
        # req = ResetFilterRequest()
        # req.time_to_wait = 0
        # req.filter_time.set(0, 0)
        # try:
        #     self.reset_proxy.call(req)
        # except rospy.ServiceException:
        #     rospy.logerr('Could not reset filter!')

if __name__ == '__main__':
    rospy.init_node('episode_terminator')
    et = EpisodeTerminator()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
