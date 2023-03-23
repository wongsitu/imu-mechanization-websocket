import numpy as np
from collections import deque


class LiveFilter:
    '''
    Base class for live filters
    '''

    def process(self, x: float) -> np.ndarray:
        # Do not process NaNs
        if np.isnan(x):
            return x
        return self._process(x)

    def __call__(self, x: float) -> np.ndarray:
        return self.process(x)

    def _process(self, x):
        raise NotImplementedError("Derived class must implement _process")


class LiveSosFilter(LiveFilter):
    '''
    Live implementation of digital filter with second-order sections
    '''

    def __init__(self, sos: np.ndarray) -> None:
        '''
        Initialize live second-order sections filter
        '''
        self.sos = sos
        self.n_sections = sos.shape[0]
        self.state = np.zeros((self.n_sections, 2))

    def _process(self, x: float) -> float:
        '''
        Filter incoming data with cascaded second-order sections
        '''
        for s in range(self.n_sections):  # Apply filter sections in sequence
            b0, b1, b2, _, a1, a2 = self.sos[s, :]

            # Compute difference equations of transposed direct form II
            y = b0 * x + self.state[s, 0]
            self.state[s, 0] = b1 * x - a1 * y + self.state[s, 1]
            self.state[s, 1] = b2 * x - a2 * y
            x = y  # Set biquad output as input of next filter section.

        return y


class MultidimensionalLiveSosFilter:
    '''
    Multidimensional extension to the LiveSosFilter class
    '''

    def __init__(self, sos: np.ndarray, shape: int | tuple[int]) -> None:
        if isinstance(shape, int):
            self.filters = [LiveSosFilter(sos) for _ in range(shape)]
        elif isinstance(shape, tuple | list):
            self.filters = [LiveSosFilter(sos) for _ in range(np.prod(shape))]
        else:
            raise ValueError(f'Input shape must be an integer or a tuple or list of integers, not {type(shape)}')
        self.shape = shape

    def process(self, x: np.ndarray) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            print('WARNING: input converted to np.ndarray in MultidimensionalLiveSosFilter')

        if isinstance(self.shape, int):
            if len(x) != self.shape:
                raise ValueError(f'Invalid input of shape {x.shape} for filter of size {self.shape}')
            return np.array([self.filters[i].process(x[i]) for i in range(self.shape)])

        if x.shape != self.shape:
            raise ValueError(f'Invalid input of shape {x.shape} for filter of shape {self.shape}')

        x = x.flatten()
        return np.array([self.filters[i].process(x[i]) for i in range(len(x))]).reshape(self.shape)


class LiveMeanFilter(LiveFilter):
    '''
    Efficient moving average filter
    '''

    def __init__(self, n: int | None = None) -> None:
        '''
        Args:
            n: int | None - Filter computes the mean of the n most recent values. 
                If n == None, all values are considered.
        '''
        self.vals = deque(maxlen=n)
        self.n = n
        self.n_inv = 1 / n
        self.k = 0
        self.mean = 0

    def _process(self, x: float) -> float:
        '''
        Update the simple moving average
        '''
        if self.k < self.n:
            self.mean = self.mean * self.k + x
            self.k += 1
            self.mean /= self.k
            self.vals.append(x)
            return self.mean

        self.mean += self.n_inv * (x - self.vals[0])
        self.vals.append(x)
        return self.mean

    def __getitem__(self, i: int) -> float:
        if i > 0:
            raise IndexError('Moving mean can only be indexed from the right (i.e. with negative indices) or 0')
        return self.vals[i]

    def __len__(self) -> int:
        return len(self.vals)

    def get_mean(self):
        '''
        Return the current mean
        '''
        return self.mean
