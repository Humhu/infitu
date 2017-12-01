"""Interfaces for synchronization frontends, and memory backends.
"""

import abc

class SynchronizationFrontend(object):
    """Interface for classes that receive and group raw data. Frontends define
    the state, action, and rewards for an application.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, backend):
        self._backend = backend

    def spin(self, current_time):
        """Process internal buffers to group and produce SARS tuples.

        Parameters
        ----------
        current_time : float
            The current time in seconds since the epoch
        """
        sars, terms = self.spin_impl(current_time)
        print '%d sars, %d terms' % (len(sars), len(terms))
        self._backend.report_sars(sars)
        self._backend.report_terminals(terms)

    @abc.abstractmethod
    def spin_impl(self, current_time):
        """Derived implementation of internal spin.
        
        Returns
        -------
        sars  : list of SARS tuples
        terms : list of SA tuples
        """
        pass
