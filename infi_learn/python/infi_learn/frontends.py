"""Synchronization frontend base class
"""
import abc
import rospy
import adel as ad

from argus_utils import TimeSeries
from infi_msgs.msg import RewardStamped, EpisodeBreak
from broadcast.msg import FloatVectorStamped

class SynchronizationFrontend(object):
    """Interface for classes that receive and group raw data. Frontends define
    the state, action, and rewards for an application.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, backend):
        self._backend = backend

    def spin(self, current_time):
        """Process internal buffers to group and produce SARS tuples.

        Parameters
        ----------
        current_time : float
            The current time in seconds since the epoch
        """
        sars, terms = self.spin_impl(current_time)
        self._backend.report_sars(sars)
        self._backend.report_terminals(terms)

    @abc.abstractmethod
    def spin_impl(self, current_time):
        """Derived implementation of internal spin.
        
        Returns
        -------
        sars  : list of SARS tuples
        terms : list of SA tuples
        """
        pass

class SARSFrontend(SynchronizationFrontend):
    """Frontend that synchronizes an data source with a
    reward signal, action broadcast topic, and episode break topic.
    """

    def __init__(self, source, backend):
        SynchronizationFrontend.__init__(self, backend)
        self.source = source

        self.action_dim = rospy.get_param('~frontend/action_dim')

        dt = rospy.get_param('~frontend/dt')
        self.lag = float(rospy.get_param('~frontend/lag'))
        tol = rospy.get_param('~frontend/sync_time_tolerance')
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
        # 1. Process data source
        head = current_time - self.lag
        for t, data in self.source.get_data(until=head):
            self.sync.buffer_state(t=t, s=data)

        # 2. Spin synchronizer
        rospy.loginfo('Sync buffer: %d states, %d actions, %d rewards',
                      self.sync.num_states_buffered,
                      self.sync.num_actions_buffered,
                      self.sync.num_rewards_buffered)
        sars, terms = self.sync.process(now=head)
        return sars, terms

    def action_callback(self, msg):
        if not self.inited:
            self.sync.buffer_episode_active(msg.header.stamp.to_sec())
            self.inited = True
        self.sync.buffer_action(t=msg.header.stamp.to_sec(), a=msg.values)

    def reward_callback(self, msg):
        self.sync.buffer_reward(t=msg.header.stamp.to_sec(), r=msg.reward)
