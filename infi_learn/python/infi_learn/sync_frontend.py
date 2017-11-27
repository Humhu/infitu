"""Synchronization frontend base class
"""
import rospy
import adel as ad

from argus_utils import TimeSeries
from infi_msgs.msg import RewardStamped, EpisodeBreak
from broadcast.msg import FloatVectorStamped

from interfaces import SynchronizationFrontend



class DataSourceFrontend(SynchronizationFrontend):
    """Frontend that synchronizes an data source with a
    reward signal, action broadcast topic, and episode break topic.
    """

    def __init__(self, source, backend):
        super(DataSourceFrontend, self).__init__(backend)
        self.source = source

        self.action_dim = rospy.get_param('~frontend/action_dim')

        dt = rospy.get_param('~frontend/dt')
        self.lag = float(rospy.get_param('~frontend/lag'))
        tol = rospy.get_param('~frontend/time_tolerance')
        self.sync = ad.SARSSynchronizer(dt=dt, tol=tol)

        # HACK to catch missing first episode start?
        self.inited = False

        self.reward_sub = rospy.Subscriber('reward', RewardStamped,
                                           callback=self.reward_callback)
        self.action_sub = rospy.Subscriber('action', FloatVectorStamped,
                                           callback=self.action_callback)
        self.break_sub = rospy.Subscriber('breaks', EpisodeBreak,
                                          callback=self.break_callback)

    def break_callback(self, msg):
        self.inited = True
        if msg.start:
            self.sync.buffer_episode_active(msg.time.to_sec())
        else:
            self.sync.buffer_episode_terminate(msg.time.to_sec())

    def spin_impl(self, current_time):
        # while len(self.image_buffer) > 0:
        #     head = self.image_buffer[0].header.stamp.to_sec()
        #     if head > current_time - self.lag:
        #         break
        #     img = self.image_buffer.popleft()

        #     bt, beliefs = self.belief_buffer.get_closest_either(head)
        #     if head - bt > self.belief_dt_tol:
        #         continue

        # 1. Process data source
        head = current_time - self.lag
        for t, data in self.source.get_data(until=head):
            self.sync.buffer_state(t=t, s=data)

        # 2. Spin synchronizer
        sars, terms = self.sync.process(now=head)
        return sars, terms

    def action_callback(self, msg):
        if not self.inited:
            self.sync.buffer_episode_active(msg.header.stamp.to_sec())
            self.inited = True
        self.sync.buffer_action(t=msg.header.stamp.to_sec(), a=msg.values)

    def reward_callback(self, msg):
        self.sync.buffer_reward(t=msg.header.stamp.to_sec(), r=msg.reward)
