"""Interfaces for synchronization frontends, and memory backends.
"""

import abc

class SynchronizationFrontend(object):
    """Interface for classes that receive and group raw data. Frontends define
    the state, action, and rewards for an application.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, backend):
        if not isinstance(backend, MemoryBackend):
            raise ValueError('backend must be implement the MemoryBackend interface')
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
        self._backend.report_sars_tuples(sars)
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

class MemoryBackend(object):
    """Interface for classes that store and retrieve learning data.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def report_sars_tuples(self, sars):
        """Reports SARS tuples from a front end.

        Parameters
        ----------
        sars : iterable of 4-tuples
            The tuples to be added to memory
        """

    @abc.abstractmethod
    def report_terminals(self, sa):
        """Reports terminal SA tuples from a front end.

        Parameters
        ----------
        sa : iterable of 2-tuples
            The tuples to be added to memory
        """