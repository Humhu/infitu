"""Memory backend class implementations.
"""
import adel
import rospy
import dill

class MonolithicBackend(object):
    """Stores all data in a single monolithic SARS dataset.
    """

    def __init__(self):
        self.training = adel.SARSDataset()
        self.validation = adel.SARSDataset()

        self.save_path = rospy.get_param('~backend/save_path', None)
        if self.save_path is not None:
            # Make sure file path is ok
            save_file = open(self.save_path, 'w')
            save_file.close()

    def __del__(self):
        if self.save_path is not None:
            dill.dump(self, open(self.save_path, 'w'))

    def report_sars(self, sars):
        for item in sars:
            self.training.report_sars(*item)

    def report_terminals(self, sa):
        for item in sa:
            self.training.report_terminal(*item)

# TODO Figure out how to save all the splits


class ActionSplitBackend(object):
    """Splits data into separate datasets for each action seen.
    """

    def __init__(self):
        self.datasets = {}

        holdout_rate = rospy.get_param('~backend/holdout_rate', 0.0)
        holdout_mode = rospy.get_param('~backend/holdout_mode', 'uniform')
        if holdout_mode == 'uniform':
            self.make_sampler = lambda: adel.UniformSampler(rate=holdout_rate)
        elif holdout_mode == 'contiguous':
            slen = rospy.get_param('~backend/holdout_segment_length')
            self.make_sampler = lambda: adel.ContiguousSampler(rate=holdout_rate,
                                                               segment_len=slen)
        else:
            raise ValueError('Unrecognized holdout mode: %s' % holdout_mode)

        self.save_path = rospy.get_param('~backend/save_path', None)
        if self.save_path is not None:
            # Make sure file path is ok
            save_file = open(self.save_path, 'w')
            save_file.close()

    def __del__(self):
        if self.save_path is not None:
            dill.dump(self, open(self.save_path, 'w'))

    def create_split(self, a):
        if a in self.datasets:
            raise ValueError('Action %s already exists' % str(a))
        
        # NOTE This simplifies the number of references we need to keep
        self.datasets[a] = adel.ValidationHoldout(training=adel.SARSDataset(),
                                                  holdout=adel.SARSDataset(),
                                                  sampler=self.make_sampler())

    def get_splits(self):
        return list(self.datasets.iterkeys())

    def get_training(self, a):
        """Returns the corresponding training data
        """
        if a not in self.datasets:
            raise ValueError('No split %s' % str(a))
        return self.datasets[a].training

    def get_validation(self, a):
        """Returns the corresponding validation data
        """
        if a not in self.datasets:
            raise ValueError('No split %s' % str(a))
        return self.datasets[a].holdout

    def report_sars(self, sars):
        for s, a, r, sn in sars:
            if a not in self.datasets:
                self.create_split(a)
            self.datasets[a].report_sars(s, a, r, sn)

    def report_terminals(self, sa):
        for s, a in sa:
            if a not in self.datasets:
                self.create_split(a)
            self.datasets[a].report_terminal(s, a)
