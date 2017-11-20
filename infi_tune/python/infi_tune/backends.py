"""Memory backend class implementations.
"""
import adel
import rospy
from interfaces import MemoryBackend


class MonolithicBackend(MemoryBackend):
    """Stores all data in a single monolithic SARS dataset.
    """

    def __init__(self):
        self.dataset = adel.EpisodicSARSDataset()

        self.save_path = rospy.get_param('~backend/save_path', None)
        if self.save_path is not None:
            # Make sure file path is ok
            save_file = open(self.save_path, 'w')
            save_file.close()

    def __del__(self):
        if self.save_path is not None:
            self.dataset.save(self.save_path)

    def report_sars_tuples(self, sars):
        for item in sars:
            self.dataset.report_sars(*item)

    def report_terminals(self, sa):
        for item in sa:
            self.dataset.report_terminal(*item)

# TODO Figure out how to save all the splits
class ActionSplitBackend(MemoryBackend):
    """Splits data into separate datasets for each action seen.
    """

    def __init__(self):
        self.datasets = {}
        self.validations = {}

        self.online_val = rospy.get_param('~backend/online_validation_rate', 0.0)

    def create_split(self, a):
        if a in self.datasets:
            raise ValueError('Action %s already exists' % str(a))
        self.datasets[a] = adel.EpisodicSARSDataset()
        self.validations[a] = adel.EpisodicSARSDataset()
        self.datasets[a].link_validation(self.validations[a])
        self.datasets[a].set_online_validation(self.online_val)

    def report_sars_tuples(self, sars):
        for s, a, r, sn in sars:
            if a not in self.datasets:
                self.create_split(a)
            self.datasets[a].report_sars(s, a, r, sn)

    def report_terminals(self, sa):
        for s, a in sa:
            if a not in self.datasets:
                self.create_split(a)
            self.datasets[a].report_terminal(s, a)
