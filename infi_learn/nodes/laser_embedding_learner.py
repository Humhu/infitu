#! /usr/bin/env python

import adel
import infi_learn as rr
import rospy
import tensorflow as tf
import numpy as np
import dill

class LaserEmbeddingFrontend(object):
    def __init__(self, backend):
        belief_dim = rospy.get_param('~frontend/belief_dim')
        self.action_dim = rospy.get_param('~frontend/action_dim')
        laser_dim = rospy.get_param('~frontend/laser_dim')
        laser_fov = rospy.get_param('~frontend/laser_fov')
        max_range = rospy.get_param('~frontend/laser_max_range')
        resolution = rospy.get_param('~frontend/laser_paint_resolution')
        dt_tol = rospy.get_param('~frontend/time_tolerance')

        self.belief_source = rr.VectorSource(dim=belief_dim, topic='belief')
        self.laser_source = rr.LaserSource(laser_dim=laser_dim, topic='scan',
                                           enable_painting=True, fov=laser_fov,
                                           max_range=max_range, resolution=resolution,
                                           enable_vis=True)
        self.state_source = rr.MultiDataSource([self.belief_source, self.laser_source],
                                               tol=dt_tol)

        self.frontend = rr.DataSourceFrontend(source=self.state_source,
                                              backend=backend)

    def get_plottables(self):
        return [self.laser_source]

    def spin(self, t):
        self.frontend.spin(t)

    @property
    def img_size(self):
        return self.laser_source.painter.img_size

    @property
    def belief_size(self):
        return self.belief_source.dim


class LaserEmbeddingLearner(rr.BaseEmbeddingLearner):
    """Combines a laser frontend with an action-split backend to learn an embedding
    for each action-split dataset.
    """

    def __init__(self):
        super(LaserEmbeddingLearner, self).__init__()

    def create_model(self, scope):
        return rr.EmbeddingModel(img_size=self.frontend.img_size,
                                 vec_size=self.frontend.belief_size,
                                 scope=scope,
                                 spec=self.network_spec)


if __name__ == '__main__':
    rospy.init_node('laser_embedding_learner')
    lel = LaserEmbeddingLearner()
    try:
        lel.spin_plot()
    except rospy.ROSInterruptException:
        pass
