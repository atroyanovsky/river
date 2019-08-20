cdef class Statistic:

    cpdef double get(self):
        """Returns the current value of the statistic."""
        raise NotImplementedError

    def __str__(self):
        return f'{self.__class__.__name__}: {self.get():.6f}'.rstrip('0')

    def __repr__(self):
        return str(self)


cdef class Univariate(Statistic):

    cpdef Univariate update(self, double x):
        """Updates and returns the called instance."""
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__.lower()


cdef class Bivariate(Statistic):

    cpdef Bivariate update(self, double x, double y):
        """Updates and returns the called instance."""
        raise NotImplementedError