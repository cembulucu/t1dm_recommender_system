import numpy as np


class UniformRandomBandit:
    """ A strategy that pulls arms uniformly at random """
    def __init__(self, da):
        self.da = da

    def determine_arm_one_round(self):
        return np.random.rand(self.da)
