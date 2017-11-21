#!/usr/bin/env python

from itertools import izip
import numpy as np
import rospy
from percepto_msgs.srv import GetCritique, GetCritiqueResponse
from argus_utils import wait_for_service
from optim import CritiqueInterface


class SamplingEvaluator:
    '''Samples an evaluator multiple times and returns statistics
    '''

    def __init__(self):

        critique_topic = rospy.get_param('~critic_service')
        self.interface = CritiqueInterface(critique_topic)
        self.num_trials = rospy.get_param('~num_trials')
        self.server = rospy.Service('~get_critique', GetCritique, self.critique_callback)
        
        self.reduce_mode = rospy.get_param('~reduce_mode', 'mean')
        self._reduce([0,0]) # Testing validity

    def _reduce(self, critiques):
        if self.reduce_mode == 'mean':
            return np.mean(critiques)
        elif self.reduce_mode == 'min':
            return np.min(critiques)
        elif self.reduce_mode == 'max':
            return np.max(critiques)
        else:
            raise ValueError('Unknown reduce mode %s' % self.reduce_mode)

    def critique_callback(self, req):

        # Perform all the requisite service calls
        try:
            responses = [self.interface.raw_call(req) for _ in range(self.num_trials)]
        except rospy.ServiceException:
            rospy.logerr('Error during critique query.')
            return None

        # Sort the responses
        critiques = []
        feedbacks = {}
        for res in responses:
            critiques.append(res.critique)
            for (fb_name, fb_val) in izip(res.feedback_names, res.feedback_values):
                if fb_name not in feedbacks:
                    feedbacks[fb_name] = []
                feedbacks[fb_name].append(fb_val)

        # Parse the responses, reporting means and variances
        res = GetCritiqueResponse()
        res.critique = self._reduce(criiques)

        res.feedback_names.append('critique_var')
        crit_var = np.var(critiques)
        res.feedback_values.append(crit_var)

        for (fb_name, fb_vals) in feedbacks.iteritems():
            res.feedback_names.append(fb_name + '_mean')
            res.feedback_values.append(np.mean(fb_vals))
            res.feedback_names.append(fb_name + '_var')
            res.feedback_values.append(np.var(fb_vals))

        # Print results
        outstr = 'critique: %f (%f)\n' % (res.critique, crit_var)
        for (fb_name, fb_val) in izip(res.feedback_names, res.feedback_values):
            outstr += '%s: %f\n' % (fb_name, fb_val)
        rospy.loginfo(outstr)

        return res


if __name__ == '__main__':
    rospy.init_node('multi_trial_evaluator')
    try:
        mte = SamplingEvaluator()
        rospy.spin()
    except ROSInterruptException:
        pass
