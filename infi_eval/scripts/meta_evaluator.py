#!/usr/bin/env python

from itertools import izip
import numpy as np
import rospy

from percepto_msgs.srv import GetCritique, GetCritiqueResponse
from optim import CritiqueInterface
from argus_utils import wait_for_service

class MetaEvaluator:
    '''Wraps multiple evaluators into a single evaluation call
    '''

    def __init__(self):
        self.critique_mode = rospy.get_param('~critique_mode')
        if self.critique_mode not in ['average']:
            raise ValueError('Unsupported critique mode: %s' % self.critique_mode)

        self.feedback_mode = rospy.get_param('~feedback_mode')
        if self.feedback_mode not in ['append']:
            raise ValueError('Unsupported feedback mode: %s' % self.feedback_mode)

        self.log_reward = rospy.get_param('~using_log_rewards')

        evaluator_info = dict(rospy.get_param('~evaluators'))
        self.evaluators = {}
        for name, topic in evaluator_info.iteritems():
            if name in self.evaluators:
                raise ValueError('Evaluator %s already registered!' % name)
            rospy.loginfo('Registering evaluator %s with topic %s',
                          name, topic)

            wait_for_service(topic)
            self.evaluators[name] = CritiqueInterface(topic)

        self.critique_server = rospy.Service('~get_critique',
                                             GetCritique,
                                             self.critique_callback)

    def critique_callback(self, req):
        results = self.run_trials(req)
        if results is None:
            return None

        critiques, feedbacks = zip(*results)
        res = GetCritiqueResponse()
        if self.critique_mode == 'average':
            if self.log_reward:
                res.critique = np.log(np.mean(np.exp(critiques)))
            else:
                res.critique = np.mean(critiques)

        if self.feedback_mode == 'append':
            for (fb, name) in izip(feedbacks, self.evaluators.iterkeys()):
                for k, v in fb.iteritems():
                    res.feedback_names.append('%s_%s' % (k, name))
                    res.feedback_values.append(v)

        return res

    def run_trials(self, req):
        results = []
        for name, proxy in self.evaluators.iteritems():
            rospy.loginfo('Evaluating on %s...', name)
            
            res = proxy(req)
            if res is None:
                rospy.logerr('Could not evaluate %s!', name)
                return None
            else:
                results.append(res)
        return results

if __name__ == '__main__':
    rospy.init_node('multi_evaluator')
    try:
        epe = MetaEvaluator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
