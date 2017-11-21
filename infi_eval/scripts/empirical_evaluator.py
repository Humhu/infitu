#!/usr/bin/env python

from threading import Lock
import rospy
from percepto_msgs.srv import GetCritique, GetCritiqueResponse,
from infi_msgs.srv import SetParameters, SetParametersRequest,
from infi_msgs.srv import StartEvaluation, StartSetup, StartTeardown, SetRecording
from fieldtrack.srv import ResetFilter, ResetFilterRequest
from argus_utils import wait_for_service


class EmpiricalEvaluator:
    '''Evaluates configurations empirically with trials

    Trial procedure
    ---------------
    1. External call to ~get_critique
    2. Evaluator runs setup service, if any
    3. Evaluator resets filter
    4. Evaluator waits for ~evaluation_delay seconds
    5. Evaluator starts recording for each registered recorder
    6. Evaluator waits for trial termination condition (see below)
    7. Evaluator stops recording for each registered recorder
    8. Evaluator calls teardown service, if any

    Trial termination
    -----------------
    The condition to use is specified with ~evaluation_mode, which can be set to:
    * service_call:   Evaluator terminates as soon as ~start_evaluation_service 
                      returns
    * fixed_duration: Evaluator terminates after ~evaluation_time seconds
    * interactive:    Evaluator terminates after receiving keypress from user
    '''

    def __init__(self):
        self.mutex = Lock()
        self.verbose = rospy.get_param('~verbose', False)

        # Create parameter setter proxy
        setter_topic = rospy.get_param('~parameter_set_service')
        wait_for_service(setter_topic)
        self.setter_proxy = rospy.ServiceProxy(setter_topic,
                                               SetParameters,
                                               True)

        # Parse evaluation mode
        self.eval_mode = rospy.get_param('~evaluation_mode')
        if self.eval_mode == 'service_call':
            evaluation_topic = rospy.get_param('~start_evaluation_service',
                                               None)
            wait_for_service(evaluation_topic)
            self.evaluation_proxy = rospy.ServiceProxy(evaluation_topic,
                                                       StartEvaluation,
                                                       True)
        elif self.eval_mode == 'fixed_duration':
            self.evaluation_time = rospy.get_param('~evaluation_time')
        elif self.eval_mode == 'interactive':
            pass
        else:
            raise ValueError('Unknown mode %s' % self.eval_mode)

        # Check for evaluation setup
        self.setup_proxy = None
        setup_topic = rospy.get_param('~start_setup_service', None)
        if setup_topic is not None:
            wait_for_service(setup_topic)
            self.setup_proxy = rospy.ServiceProxy(setup_topic,
                                                  StartSetup,
                                                  True)

        # Check for evaluation teardown
        self.teardown_proxy = None
        teardown_topic = rospy.get_param('~start_teardown_service', None)
        if teardown_topic is not None:
            wait_for_service(teardown_topic)
            self.teardown_proxy = rospy.ServiceProxy(teardown_topic,
                                                     StartTeardown,
                                                     True)

        # Create filter reset proxy
        self.reset_proxy = None
        reset_topic = rospy.get_param('~reset_filter_service', None)
        if reset_topic is not None:
            wait_for_service(reset_topic)
            self.reset_proxy = rospy.ServiceProxy(reset_topic,
                                                  ResetFilter,
                                                  True)

        # Register recorders
        recording_topics = dict(rospy.get_param('~recorders'))
        self.recorders = {}
        for name, topic in recording_topics.iteritems():
            wait_for_service(topic)
            self.recorders[name] = rospy.ServiceProxy(topic,
                                                      SetRecording,
                                                      True)
        self.critique_record = rospy.get_param('~critique_record')
        if self.critique_record not in self.recorders:
            raise ValueError('Critique not a registered recorder!')

        # Parse eval delay
        eval_delay = rospy.get_param('~evaluation_delay', 0.0)
        self.evaluation_delay = rospy.Duration(eval_delay)

        # Create critique server
        self.critique_server = rospy.Service('~get_critique',
                                             GetCritique,
                                             self.critique_callback)

    def start_recording(self):
        '''Start evaluation recording. Returns success.
        '''
        try:
            for recorder in self.recorders.itervalues():
                recorder.call(True)
        except rospy.ServiceException:
            rospy.logerr('Could not start recording.')
            return False
        return True

    def stop_recording(self):
        '''Stop recording and return evaluation. Returns None if fails.
        '''
        try:
            feedback = []
            for name, recorder in self.recorders.iteritems():
                res = recorder.call(False).evaluation
                feedback.append((name, res))
                if name == self.critique_record:
                    critique = res
        except rospy.ServiceException:
            rospy.logerr('Could not stop recording.')
            return None

        return (critique, feedback)

    def set_parameters(self, inval):
        '''Set the parameters to be evaluated. Returns success.
        '''
        preq = SetParametersRequest()
        preq.parameters = inval
        try:
            self.setter_proxy.call(preq)
        except rospy.ServiceException:
            rospy.logerr('Could not set parameters to %s', str(inval))
            return False
        return True

    def reset_filter(self):
        '''Reset the state estimator. Returns success.
        '''
        if self.reset_proxy is None:
            return True

        resreq = ResetFilterRequest()
        resreq.time_to_wait = 0
        resreq.filter_time = rospy.Time.now()
        try:
            self.reset_proxy.call(resreq)
        except rospy.ServiceException:
            rospy.logerr('Could not reset filter.')
            return False
        return True

    def wait_for_termination(self):
        '''Waits for trial to terminate
        '''
        if self.eval_mode == 'service_call':
            try:
                self.evaluation_proxy.call()
            except rospy.ServiceException:
                rospy.logerr('Could not run evaluation.')
                return False

        elif self.eval_mode == 'fixed_duration':
            rospy.sleep(self.evaluation_time)

        elif self.eval_mode == 'interactive':
            rospy.loginfo('Waiting on user input to end evaluation...')
            raw_input('Press a key to end evaluation...')

        return True

    def start_setup(self):
        '''Perform trial setup
        '''
        if self.eval_mode == 'interactive':
            rospy.loginfo('Waiting on user input to begin evaluation...')
            raw_input('Press a key to begin evaluation...')

        if self.setup_proxy is None:
            return True

        try:
            self.setup_proxy.call()
        except rospy.ServiceException:
            rospy.logerr('Could not setup.')
            return False
        return True

    def start_teardown(self):
        '''Perform trial teardown
        '''
        if self.teardown_proxy is None:
            return True

        try:
            self.teardown_proxy.call()
        except rospy.ServiceException:
            rospy.logerr('Could not teardown')
            return False

        return True

    def critique_callback(self, req):
        with self.mutex:
            if not self.start_setup():
                return None

            # Call parameter setter
            if not self.set_parameters(req.input):
                self.start_teardown()
                return None

            # Reset state estimator
            if not self.reset_filter():
                self.start_teardown()
                return None

            # Wait before starting
            # NOTE This catches instances when relying on sim time
            if self.evaluation_delay.to_sec() > 0:
                rospy.sleep(self.evaluation_delay)

            if not self.start_recording():
                self.start_teardown()
                return None

            # Wait until evaluation is done
            if not self.wait_for_termination():
                self.start_teardown()
                return None

            # Get outcomes
            res = GetCritiqueResponse()
            records = self.stop_recording()
            if records is None:
                self.start_teardown()
                return None

            res.critique, feedback = records
            res.feedback_names = [f[0] for f in feedback]
            res.feedback_values = [f[1] for f in feedback]

            if not self.start_teardown():
                return None

            return res


if __name__ == '__main__':
    rospy.init_node('empirical_evaluator')
    try:
        epe = EmpiricalEvaluator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
