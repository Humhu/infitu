"""Classes that sort data into datasets
"""
import adel
import dill

# TODO Implement same interface as keysplitbackend?
class MonolithicBackend(object):
    """Stores all data in a single monolithic SARS dataset.
    """

    def __init__(self, save_path=None):
        self.training = adel.SARSDataset()
        self.validation = adel.SARSDataset()

        self.save_path = save_path
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

def strict_state_keyfunc(*args):
    """Keyfunc for splitting data based on state
    """
    if len(args) == 2 or len(args) == 4:
        return args[0]
    else:
        raise ValueError('Invalid number of arguments: %d' % len(args))


def strict_action_keyfunc(*args):
    """Keyfunc for splitting data based on action taken
    """
    if len(args) == 2 or len(args) == 4:
        return args[1]
    else:
        raise ValueError('Invalid number of arguments: %d' % len(args))


# TODO Figure out how to save all the splits
class KeySplitBackend(object):
    """Splits data into separate datasets based on a SARS key function.
    Automatically creates training/holdout subsets for each split.
    """

    def __init__(self, keyfunc, validation, save_path=None):
        self.keyfunc = keyfunc
        self.datasets = {}
        self.validation_args = validation

        self.save_path = save_path
        if self.save_path is not None:
            # Make sure file path is ok
            save_file = open(self.save_path, 'w')
            save_file.close()

    def __del__(self):
        if self.save_path is not None:
            dill.dump(self, open(self.save_path, 'w'))

    def create_split(self, k):
        if k in self.datasets:
            raise ValueError('Action %s already exists' % str(k))
        
        self.datasets[k] = adel.ValidationHoldout(training=adel.SARSDataset(),
                                                  holdout=adel.SARSDataset(),
                                                  sampler=adel.create_sampler(**self.validation_args))

    def get_splits(self):
        return list(self.datasets.iterkeys())

    def get_training(self, k):
        """Returns the corresponding training data
        """
        if k not in self.datasets:
            raise ValueError('No split %s' % str(k))
        return self.datasets[k].training

    def get_validation(self, k):
        """Returns the corresponding validation data
        """
        if k not in self.datasets:
            raise ValueError('No split %s' % str(k))
        return self.datasets[k].holdout

    def report_sars(self, sars):
        for item in sars:
            k = self.keyfunc(*item)
            if k not in self.datasets:
                self.create_split(k)
            self.datasets[k].report_sars(*item)

    def report_terminals(self, sa):
        for item in sa:
            k = self.keyfunc(*item)
            if k not in self.datasets:
                self.create_split(k)
            self.datasets[k].report_terminal(*item)
