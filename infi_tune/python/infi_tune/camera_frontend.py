"""Image frontend class
"""

from collections import deque

import rospy
import adel as ad
import numpy as np

from argus_utils import TimeSeries
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from percepto_msgs.msg import RewardStamped, EpisodeBreak
from broadcast.msg import FloatVectorStamped

from relearn_reconfig.interfaces import SynchronizationFrontend
from relearn_reconfig.utils import LaserImagePainter

import matplotlib.pyplot as plt


class CameraFrontend(SynchronizationFrontend):
    """Camera data frontend that synchronizes an image topic,
    vector broadcast topic, reward signal, action broadcast topic,
    and episode break topic.
    """
    def __init__(self, backend):
        super(CameraFrontend, self).__init__(backend)

        self.image_dims = rospy.get_param('~frontend/image_dims')
        self.belief_dim = rospy.get_param('~frontend/belief_dim')
        self.action_dim = rospy.get_param('~frontend/action_dim')

        dt = rospy.get_param('~frontend/dt')
        self.lag = float(rospy.get_param('~frontend/lag'))
        tol = rospy.get_param('~frontend/time_tolerance')
        self.sync = ad.SARSSynchronizer(dt=dt, lag=self.lag, tol=tol)

        self.belief_dt_tol = rospy.get_param('~frontend/belief_dt_tol')

        self.belief_buffer = TimeSeries()
        self.belief_sub = rospy.Subscriber('belief', FloatVectorStamped,
                                           callback=self.belief_callback)
        self.image_buffer = deque()

        # HACK to catch missing first episode start?
        self.inited = False

        self.image_mode = rospy.get_param('~frontend/image_mode', 'grayscale')
        self.num_image_channels() # NOTE Arg checking
        self.image_bridge = CvBridge()
        self.camera_sub = rospy.Subscriber('image', Image,
                                          callback=self.image_callback)
        self.reward_sub = rospy.Subscriber('reward', RewardStamped,
                                           callback=self.reward_callback)
        self.action_sub = rospy.Subscriber('action', FloatVectorStamped,
                                           callback=self.action_callback)
        self.break_sub = rospy.Subscriber('breaks', EpisodeBreak,
                                          callback=self.break_callback)

    def num_image_channels(self):
        if self.image_mode == 'gray':
            return 1
        elif self.image_mode == 'rgb':
            return 3
        else:
            raise ValueError('Invalid image mode: %s' % self.image_mode)

    def break_callback(self, msg):
        self.inited = True
        if msg.start:
            self.sync.buffer_episode_active(msg.time.to_sec())
        else:
            self.sync.buffer_episode_terminate(msg.time.to_sec())

    def belief_callback(self, msg):
        self.belief_buffer.insert(msg.header.stamp.to_sec(), msg.values)

    def spin_impl(self, current_time):
        # 1. Process image buffer
        while len(self.image_buffer) > 0:
            head = self.image_buffer[0].header.stamp.to_sec()
            if head > current_time - self.lag:
                break
            img = self.image_buffer.popleft()

            bt, beliefs = self.belief_buffer.get_closest_either(head)
            if head - bt > self.belief_dt_tol:
                continue

            self.sync.buffer_state(t=head, s=(beliefs, img))

        # 2. Process synchronizer
        sars, terms = self.sync.process(now=current_time)
        return sars, terms

    def action_callback(self, msg):
        if not self.inited:
            self.sync.buffer_episode_active(msg.header.stamp.to_sec())
            self.inited = True
        self.sync.buffer_action(t=msg.header.stamp.to_sec(), a=msg.values)

    def image_callback(self, msg):
        if self.image_mode == 'gray':
            try:
                cv_image = self.image_bridge.imgmsg_to_cv2(msg, "mono8")
            except CvBridgeError as e:
                rospy.logerr('Could not convert image: %s', str(e))
                return
        elif self.image_mode == 'rbg':
            try:
                cv_image = self.image_bridge.imgmsg_to_cv2(msg, "rgb8")
            except CvBridgeError as e:
                rospy.logerr('Could not convert image: %s', str(e))
                return

        self.image_buffer.append(cv_image)

    def reward_callback(self, msg):
        self.sync.buffer_reward(t=msg.header.stamp.to_sec(), r=msg.reward)
