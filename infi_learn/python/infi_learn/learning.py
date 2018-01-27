"""Tensorflow learner wrapper
"""

import tensorflow as tf
import adel


class Learner(object):
    def __init__(self, variables, updates, loss, learning_rate=1e-3,
                 beta1=0.9, beta2=0.95, **kwargs):
        self.step_counter = 1
        self.global_step = tf.placeholder(dtype=tf.int64)
        self.learning_rate = tf.train.exponential_decay(global_step=self.global_step,
                                                        learning_rate=float(
                                                            learning_rate),
                                                        **kwargs)

        self.loss = loss
        self.learner = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                              beta1=beta1, beta2=beta2)

        with tf.control_dependencies(updates):
            self.train = self.learner.minimize(loss, var_list=variables)

        self.init = [adel.optimizer_initializer(
            self.learner, var_list=variables)]

    def step(self, sess, feed):
        feed[self.global_step] = self.step_counter
        self.step_counter += 1
        return sess.run([self.loss, self.train], feed_dict=feed)[0]
