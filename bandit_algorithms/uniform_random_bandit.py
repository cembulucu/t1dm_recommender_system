import numpy as np


class UniformRandomBandit:
    """ A strategy that pulls arms uniformly at random """
    def __init__(self, da):
        self.da = da

    def determine_arm_one_round(self, context):
        return np.random.rand(self.da)

    def update_statistics(self, reward):
        pass

