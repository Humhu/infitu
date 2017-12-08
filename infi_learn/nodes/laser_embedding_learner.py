#! /usr/bin/env python

import adel
import infi_learn as rr
import rospy
import tensorflow as tf
import numpy as np
import dill


class LaserDataSources(object):
    def __init__(self, backend, laser=None, belief=None, dt_tol=0.1, **kwargs):
        self.action_dim = rospy.get_param('~frontend/action_dim')

        self.use_laser = laser is not None
        if self.use_laser:
            self.laser_source = rr.LaserSource(**laser)

        belief_args = belief is not None
        self.use_belief = belief_args is not None
        if self.use_belief:
            self.belief_source = rr.VectorSource(**belief)

        if self.use_laser and self.use_belief:
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
                                        backend=backend,
                                        **kwargs)

    def get_plottables(self):
        if self.use_laser:
            return [self.laser_source]
        else:
            return []

    @property
    def img_size(self):
        if self.use_laser:
            return self.laser_source.painter.img_size
        else:
            return None

    @property
    def belief_size(self):
        if self.use_belief:
            return self.belief_source.dim
        else:
            return None


if __name__ == '__main__':
    rospy.init_node('laser_embedding_learner')

    # Learn one embedding per action
    backend = rr.KeySplitBackend(keyfunc=rr.strict_action_keyfunc,
                                 **rospy.get_param('~backend'))

    sources = LaserDataSources(backend=backend,
                               **rospy.get_param('~frontend'))
    network = rr.NetworkWrapper(**rospy.get_param('~network'))

    def make_model(scope):
        return rr.EmbeddingModel(img_size=sources.img_size,
                                 vec_size=sources.belief_size,
                                 scope=scope,
                                 spec=network.network_args)

    learner = rr.EmbeddingLearner(make_model=make_model,
                                  backend=backend,
                                  network=network,
                                  **rospy.get_param('~learner'))

    plot_group = rr.PlottingGroup()
    for p in sources.get_plottables() + learner.get_plottables():
        plot_group.add_plottable(p)

    def spin(event):
        sources.frontend.spin(event.current_real.to_sec())
        learner.spin()

    plot_rate = rospy.get_param('~plot_rate', 10.0)
    spin_rate = rospy.get_param('~spin_rate')
    spin_dt = 1.0 / spin_rate
    spin_timer = rospy.Timer(rospy.Duration(spin_dt),
                             callback=spin)

    try:
        plot_group.spin(plot_rate)
    except rospy.ROSInterruptException:
        pass
