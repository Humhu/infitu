#! /usr/bin/env python

import adel
import infi_learn as rr
import rospy
import tensorflow as tf
import numpy as np
import dill


class LaserEmbeddingFrontend(object):
    def __init__(self, backend):
        self.action_dim = rospy.get_param('~frontend/action_dim')

        self.use_laser = rospy.get_param('~frontend/use_laser', False)
        if self.use_laser:
            laser_dim = rospy.get_param('~frontend/laser_dim')
            laser_fov = rospy.get_param('~frontend/laser_fov')
            max_range = rospy.get_param('~frontend/laser_max_range')
            resolution = rospy.get_param('~frontend/laser_paint_resolution')
            self.laser_source = rr.LaserSource(laser_dim=laser_dim, topic='scan',
                                               enable_painting=True, fov=laser_fov,
                                               max_range=max_range, resolution=resolution,
                                               enable_vis=True)

        self.use_belief = rospy.get_param('~frontend/use_belief', False)
        if self.use_belief:
            self.belief_dim = rospy.get_param('~frontend/belief_dim')
            self.belief_source = rr.VectorSource(dim=self.belief_dim,
                                                 topic='belief')

        if self.use_laser and self.use_belief:
            dt_tol = rospy.get_param('~frontend/state_time_tolerance')
            # By convention (vecs, imgs)
            self.state_source = rr.MultiDataSource([self.belief_source, self.laser_source],
                                                   tol=dt_tol)
        elif self.use_laser and not self.use_belief:
            self.state_source = self.laser_source
        elif not self.use_laser and self.use_belief:
            self.state_source = self.belief_source
        else:
            raise ValueError('Must use laser and/or belief')

        self.frontend = rr.SARSFrontend(source=self.state_source,
                                        backend=backend)

    def get_plottables(self):
        if self.use_laser:
            return [self.laser_source]
        else:
            return []

    def spin(self, t):
        self.frontend.spin(t)

    @property
    def img_size(self):
        if self.use_laser:
            return self.laser_source.painter.img_size
        else:
            return None

    @property
    def belief_size(self):
        if self.use_belief:
            return self.belief_dim
        else:
            return None


class LaserEmbeddingLearner(rr.BaseEmbeddingLearner):
    """Combines a laser frontend with an action-split backend to learn an embedding
    for each action-split dataset.
    """

    def __init__(self):
        super(LaserEmbeddingLearner, self).__init__(LaserEmbeddingFrontend)

    def create_model(self, scope):
        return rr.EmbeddingModel(img_size=self.frontend.img_size,
                                 vec_size=self.frontend.belief_size,
                                 scope=scope,
                                 spec=self.trainer.network_args)


if __name__ == '__main__':
    rospy.init_node('laser_embedding_learner')
    lel = LaserEmbeddingLearner()
    try:
        lel.spin_plot()
    except rospy.ROSInterruptException:
        pass
