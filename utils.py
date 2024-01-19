import numpy as np


class LinearSchedule:
    def __init__(self, start, end, duration):
        self.end = end
        self.slope = (end - start) / duration

    def __call__(self, num_steps):
        return np.max([self.end, self.slope*num_steps])
