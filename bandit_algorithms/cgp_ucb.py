import itertools

import numpy as np

from utilities.gp_utils import rbf_kernel_w_linear_coefficients


class ContextualGaussianProcessUpperConfidenceBoundAlgorithm:
    """ Implementation of CGP-UCB """
    def __init__(self, horizon, dx, da, space_diameter, delta, arm_limits=(0, 1.00000001), arm_granularity=0.01, noise_sigma=1,
                 confidence_scale=1.0):
        self.horizon, self.dx, self.da = horizon, dx, da
        self.t, self.played_points_hist, self.rews_hist = 0, np.zeros(shape=(horizon, dx+da)), np.zeros(shape=(horizon,))
        arm_grid_1d = np.arange(arm_limits[0], arm_limits[1], arm_granularity)
        self.arm_grid = np.array(list(itertools.product(arm_grid_1d, repeat=da)))
        self.kernel_fn = rbf_kernel_w_linear_coefficients
        self.noise_sigma = noise_sigma
        self.r = space_diameter
        self.delta = delta
        self.cs = confidence_scale
        self.last_played_point = None

    def select_arm(self, context):
        if self.t == 0:
            rand_int = np.random.randint(low=0, high=self.arm_grid.shape[0])
            selected_arm = self.arm_grid[rand_int]
            self.last_played_point = np.concatenate((context, selected_arm))
        else:
            context_arm_grid = self.calculate_context_arm_grid(context=context)
            mean_est, var_est = self.calculate_posterior_mean_var(context_arm_grid=context_arm_grid)
            beta_t = self.calculate_beta_t(t=self.t)
            cgp_arm_ind, cgp_high_ucb = self.calculate_highest_ucb_index(beta_t, mean_est, var_est)
            self.last_played_point = context_arm_grid[cgp_arm_ind]
            selected_arm = context_arm_grid[cgp_arm_ind, self.dx:]
        return selected_arm

    def update_statistics(self, reward):
        self.played_points_hist[self.t] = self.last_played_point
        self.rews_hist[self.t] = reward
        self.t += 1

    def calculate_beta_t(self, t):
        """Calculate beta_t for CGP-UCB"""
        # TODO: we have beta for compact, convex spaces -> add beta for arbitrary spaces
        a, d = self.delta * np.exp(-1), self.dx + self.da
        first_term = 2 * np.log((t ** 2) * 2 * (np.pi ** 2) / (3 * self.delta))
        second_term = 2 * d * np.log((t ** 2) * d * self.r * np.sqrt(np.log(4 * d * a / self.delta)))
        return first_term + second_term

    def calculate_context_arm_grid(self, context):
        """For each arm, calculates the points when concatenated with the context"""
        contexts_repeated = np.repeat(np.expand_dims(context, axis=0), repeats=self.arm_grid.shape[0], axis=0)
        context_arm_grid = np.concatenate((contexts_repeated, self.arm_grid), axis=-1)
        return context_arm_grid

    def calculate_posterior_mean_var(self, context_arm_grid):
        """Calculates statistics of the posterior distribution of expected reward function, possibly ignoring some dimensions acc. to
        ard_coeffs """
        data = self.played_points_hist[:self.t]  # all points played so far
        kernel_vectors = self.kernel_fn(context_arm_grid, data)  # kernels between all possible context-arms and the previous rounds
        kernel_matrix = self.kernel_fn(data, data)  # kernel matrix of data
        c_matrix = kernel_matrix + (self.noise_sigma ** 2) * np.eye(data.shape[0])
        c_matrix_inv = np.linalg.inv(c_matrix)
        mu_ests_vector = np.matmul(kernel_vectors, np.matmul(c_matrix_inv, self.rews_hist[:self.t]))  # mean estimation
        sigma_ests_first_term = np.diag(self.kernel_fn(context_arm_grid, context_arm_grid))
        sigma_ests_second_term = np.diag(np.matmul(kernel_vectors, np.matmul(c_matrix_inv, kernel_vectors.T)))
        sigma_ests_vector = sigma_ests_first_term - sigma_ests_second_term  # variance estimation
        return mu_ests_vector, sigma_ests_vector

    def calculate_highest_ucb_index(self, beta, means, stds, return_multiple=False):
        """ calculate highest UCB"""
        ucbs = means + self.cs * beta * stds
        highest_indices = np.argwhere(ucbs == np.max(ucbs))
        if return_multiple:
            return highest_indices, ucbs[highest_indices]
        else:
            if highest_indices.size > 1:
                highest_index = np.random.choice(np.squeeze(highest_indices))
            else:
                highest_index = np.squeeze(highest_indices)
            return highest_index, ucbs[highest_index]
