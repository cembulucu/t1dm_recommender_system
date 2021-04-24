import itertools
import sklearn.metrics as skmetrics
import numpy as np


class InstanceBasedUniformPartitioning:
    def __init__(self, horizon, dx, da, conf_scale=1.0):
        self.horizon, self.conf_scale = horizon, conf_scale
        self.dx, self.da, self.d = dx, da, (dx + da)

        self.m = np.ceil(horizon ** (1 / (2 + self.d))).astype(int)

        # calculate centers for one dimension
        centers_vector = np.arange(self.m) / self.m + (0.5 / self.m)
        # extend centers for all dimensions
        self.centers = np.array(list(itertools.product(centers_vector, repeat=self.d)))
        self.partition_size = self.centers.shape[0]

        # get centers for contexts
        self.context_centers, self.int_centers = np.unique(self.centers[:, :self.dx], return_inverse=True, axis=0)

        self.sample_counts = np.zeros(shape=(self.partition_size, ))
        self.sample_means = np.zeros(shape=(self.partition_size, ))

        self.last_played_arm_ind = -1

    def select_arm(self, context):
        # calculate distances from context to all context centers
        dist_to_centers = np.squeeze(skmetrics.pairwise_distances(np.expand_dims(context, axis=0),
                                                                  self.context_centers, metric='cityblock'))
        # get the index of closes context center
        min_dist_ind = np.argmin(dist_to_centers)
        # from all space get partitions that has the context
        relevant_arm_inds = np.squeeze(np.argwhere(self.int_centers == min_dist_ind))
        rel_sample_means, rel_sample_counts = self.sample_means[relevant_arm_inds], self.sample_counts[relevant_arm_inds]

        # calculate confidence terms
        with np.errstate(divide='ignore'):
            inside_sqrt_1 = 2 / rel_sample_counts
            inside_sqrt_2 = 1 + 2 * np.log(2*(self.m**self.da)*self.partition_size*(self.horizon**1.5))
            rel_confidence_radii = np.sqrt(inside_sqrt_1*inside_sqrt_2)

        # calculate UCBs(scaled)
        rel_ucbs = rel_sample_means + self.conf_scale * rel_confidence_radii
        # select the best hypercube, if multiple of them are best, select randomly
        winner_rel_arm_pseudo_indices = np.argwhere(rel_ucbs == np.max(rel_ucbs))
        if winner_rel_arm_pseudo_indices.shape[0] > 1:
            winner_pseudo = np.squeeze(winner_rel_arm_pseudo_indices[np.random.randint(winner_rel_arm_pseudo_indices.shape[0], size=1)])
        else:
            winner_pseudo = np.squeeze(winner_rel_arm_pseudo_indices)
        # get original index for winner hypercube
        winner_ind = np.squeeze(relevant_arm_inds[winner_pseudo])

        # update last played arm
        self.last_played_arm_ind = winner_ind

        # select an arm uniformly random inside the hypercube
        up_bound_hi, low_bound_hi = self.centers[winner_ind] + 0.5/self.m, self.centers[winner_ind] - 0.5/self.m
        arm = np.random.uniform(low=up_bound_hi[self.dx:], high=low_bound_hi[self.dx:])
        return arm

    def update_statistics(self, reward):
        # update counters for the last played arm
        y_ind = self.last_played_arm_ind
        self.sample_means[y_ind] = (self.sample_means[y_ind]*self.sample_counts[y_ind] + reward) / (self.sample_counts[y_ind] + 1)
        self.sample_counts[y_ind] = self.sample_counts[y_ind] + 1
