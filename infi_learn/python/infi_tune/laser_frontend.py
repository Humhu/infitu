"""Laser scan frontend class
"""

from collections import deque

import rospy
import adel as ad
import numpy as np

from argus_utils import TimeSeries
from sensor_msgs.msg import LaserScan
from percepto_msgs.msg import RewardStamped, EpisodeBreak
from broadcast.msg import FloatVectorStamped

from relearn_reconfig.interfaces import SynchronizationFrontend
from relearn_reconfig.utils import LaserImagePainter

import matplotlib.pyplot as plt


class LaserFrontend(SynchronizationFrontend):
    """Laser data frontend that synchronizes a planar laser scan,
    vector broadcast topic, reward signal, action broadcast topic,
    and episode break topic.

    Optionally can convert laser scans to 2D images by enabling painting.
    """
    def __init__(self, backend):
        super(LaserFrontend, self).__init__(backend)

        self.laser_dim = rospy.get_param('~frontend/laser_dim')
        self.belief_dim = rospy.get_param('~frontend/belief_dim')
        self.action_dim = rospy.get_param('~frontend/action_dim')

        self.painter = None
        enable_painting = rospy.get_param('~frontend/enable_painting', False)
        if enable_painting:
            fov = rospy.get_param('~frontend/laser_fov')
            max_range = rospy.get_param('~frontend/laser_max_range')
            resolution = rospy.get_param('~frontend/painting_resolution')
            self.painter = LaserImagePainter(laser_dim=self.laser_dim,
                                             laser_fov=fov,
                                             max_range=max_range,
                                             resolution=resolution,
                                             dtype=np.uint8,
                                             empty_val=0,
                                             fill_val=1)
            rospy.loginfo('Enabled painting with image size: %s' % str(self.painter.img_size))
            self.enable_vis = rospy.get_param('~frontend/enable_visualization', False)
            if self.enable_vis:
                plt.ion()
                self.pfig = plt.figure()

        self.laser_nan_val = float(rospy.get_param('~frontend/laser_nan_value', -1))
        self.laser_inf_val = float(rospy.get_param('~frontend/laser_inf_value'))

        dt = rospy.get_param('~frontend/dt')
        self.lag = float(rospy.get_param('~frontend/lag'))
        tol = rospy.get_param('~frontend/time_tolerance')
        self.sync = ad.SARSSynchronizer(dt=dt, lag=self.lag, tol=tol)

        self.belief_dt_tol = rospy.get_param('~frontend/belief_dt_tol')

        self.belief_buffer = TimeSeries()
        self.belief_sub = rospy.Subscriber('belief', FloatVectorStamped,
                                           callback=self.belief_callback)
        self.scan_buffer = deque()

        # HACK to catch missing first episode start?
        self.inited = False

        self.laser_sub = rospy.Subscriber('scan', LaserScan,
                                          callback=self.scan_callback)
        self.reward_sub = rospy.Subscriber('reward', RewardStamped,
                                           callback=self.reward_callback)
        self.action_sub = rospy.Subscriber('action', FloatVectorStamped,
                                           callback=self.action_callback)
        self.break_sub = rospy.Subscriber('breaks', EpisodeBreak,
                                          callback=self.break_callback)

    def break_callback(self, msg):
        # rospy.loginfo('Received break: %s', msg.cause)
        self.inited = True
        if msg.start:
            # rospy.loginfo('Episode start: %f', msg.time.to_sec())
            self.sync.buffer_episode_active(msg.time.to_sec())
        else:
            # rospy.loginfo('Episode end: %f', msg.time.to_sec())
            self.sync.buffer_episode_terminate(msg.time.to_sec())

    def belief_callback(self, msg):
        self.belief_buffer.insert(msg.header.stamp.to_sec(), msg.values)

    def spin_impl(self, current_time):
        # 1. Process scan buffer
        while len(self.scan_buffer) > 0:
            head = self.scan_buffer[0].header.stamp.to_sec()
            if head > current_time - self.lag:
                break
            scan = self.scan_buffer.popleft()

            bt, beliefs = self.belief_buffer.get_closest_either(head)
            if head - bt > self.belief_dt_tol:
                continue
            # rospy.loginfo('Buffered scan/state at %f/%f', head, bt)
            self.sync.buffer_state(t=head, s=(beliefs, scan))

        # 2. Process synchronizer
        sars, terms = self.sync.process(now=current_time)
        return sars, terms

    def action_callback(self, msg):
        if not self.inited:
            self.sync.buffer_episode_active(msg.header.stamp.to_sec())
            self.inited = True
        self.sync.buffer_action(t=msg.header.stamp.to_sec(), a=msg.values)

    def scan_callback(self, msg):
        self.scan_buffer.append(self._proc_scan(msg))

    def reward_callback(self, msg):
        self.sync.buffer_reward(t=msg.header.stamp.to_sec(), r=msg.reward)

    def _proc_scan(self, scan):
        ranges = np.array(scan.ranges)
        ranges[np.isnan(ranges)] = self.laser_nan_val
        ranges[ranges == float('inf')] = self.laser_inf_val
        if self.painter is not None:
            ranges = self.painter.scan_to_image(ranges)
            if self.enable_vis:
                plt.figure(self.pfig.number)
                self.pfig.clear()
                plt.imshow(ranges[:,:,0])
                plt.pause(0.001)
        return ranges
