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
        
        self.start_delay = float(rospy.get_param('~start_delay', 0.0))
        self.min_episode_len = float(rospy.get_param('~min_ep_len', 0.0))
        self.episode_start = rospy.Time.now()
        self.break_detected = False
        self.break_reward = 0
        self.break_time = rospy.Time()

        self.reward_sub = rospy.Subscriber('reward', RewardStamped,
                                           queue_size=1,
                                           callback=self.reward_callback)


    def reward_callback(self, msg):
        # If episode hasn't started yet, come back later
        if msg.header.stamp < self.episode_start:
            return
        
        # Check and remember if break condition is met
        break_found = False
        if not self.break_detected and msg.reward < self.min_reward:
            rospy.loginfo('Break detected: reward %f less than threshold %f', msg.reward, self.min_reward)
            self.break_detected = True
            self.break_reward = msg.reward
            self.break_time = msg.header.stamp
            break_found = True
        
        # If no break detected, nothing to do
        if not self.break_detected:
            return

        # If min length not yet reached, come back later
        ep_age = (msg.header.stamp - self.episode_start).to_sec()
        if ep_age < self.min_episode_len:
            if break_found:
                rospy.loginfo('Episode length %f less than min %f, waiting to end episode...', ep_age, self.min_episode_len )
            return

        # At this point, we should have a detected break and min length has passed
        # Publish the end of the episode
        out = EpisodeBreak()
        out.time = msg.header.stamp
        out.start = False
        out.cause = 'Reward %f at time %s less than threshold %f' % (self.break_reward, str(self.break_time), self.min_reward)
        self.break_pub.publish(out)
        rospy.loginfo(out.cause)

        # Reset the filter
        req = ResetFilterRequest()
        req.time_to_wait = 0
        req.filter_time.set(0,0)
        try:
            self.reset_proxy.call(req)
        except rospy.ServiceException:
            rospy.logerr('Could not reset filter!')

        # Publish the start of the new episode
        out = EpisodeBreak()
        out.start = True
        if self.start_delay > 0:
            rospy.loginfo('Waiting %f seconds before starting new episode...', self.start_delay)
            rospy.sleep(self.start_delay)
            out.cause = 'Starting new episode after %f second delay' % self.start_delay
        else:
            out.cause = 'Starting new episode' 
        out.time = rospy.Time.now()
        rospy.loginfo('Starting new episode')
        self.break_pub.publish(out)
        
        # Reset detection state
        self.episode_start = out.time
        self.break_detected = False

if __name__ == '__main__':
    rospy.init_node('episode_terminator')
    et = EpisodeTerminator()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
