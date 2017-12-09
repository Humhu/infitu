"""Datastream interfacing frontend classes
"""
import abc
import rospy
import adel as ad

from argus_utils import TimeSeries
from infi_msgs.msg import RewardStamped, EpisodeBreak
from broadcast.msg import FloatVectorStamped


class BaseFrontend(object):
    """Interface for classes that interface with raw datastreams. Wraps a
    data backend to automatically queue data.

    Frontends define the state, action, and rewards for a problem.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.callbacks = []

    def register_callback(self, backend):
        self.callbacks.append(backend)

    def spin(self, current_time):
        """Process internal buffers to group and produce SARS tuples.

        Parameters
        ----------
        current_time : float
            The current time in seconds since the epoch
        """
        data = self.spin_impl(current_time)
        for c in self.callbacks:
            for key, payload in data:
                c(key, payload)

    @abc.abstractmethod
    def spin_impl(self, current_time):
        """Derived implementation of internal spin.

        Returns
        -------
        data : list of (key, payload) tuples
        """
        pass


class SARSFrontend(BaseFrontend):
    """Frontend that synchronizes an data source with a
    reward signal, action broadcast topic, and episode break topic.

    Returns data with a boolean key and tuple payload
        Active SARS data have key True and payload (s, a, r, sn)
        Terminal SA data have False and payload (s, a)
    """

    def __init__(self, source, action_dim, dt,
                 lag=1.0, sync_time_tolerance=0.1):
        BaseFrontend.__init__(self)
        self.source = source

        self.action_dim = action_dim

        self.lag = lag
        self.sync = ad.SARSSynchronizer(dt=dt, tol=sync_time_tolerance)

        self.reward_sub = rospy.Subscriber('reward', RewardStamped,
                                           callback=self.reward_callback)
        self.action_sub = rospy.Subscriber('action', FloatVectorStamped,
                                           callback=self.action_callback)
        self.break_sub = rospy.Subscriber('breaks', EpisodeBreak,
                                          callback=self.break_callback)

    def break_callback(self, msg):
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

        return [(True, i) for i in sars] + [(False, i) for i in terms]

    def action_callback(self, msg):
        self.sync.buffer_action(t=msg.header.stamp.to_sec(), a=msg.values)

    def reward_callback(self, msg):
        self.sync.buffer_reward(t=msg.header.stamp.to_sec(), r=msg.reward)
