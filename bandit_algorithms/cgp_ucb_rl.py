import itertools

import numpy as np

from utilities.gp_utils import rbf_kernel_w_linear_coefficients, calculate_power_set


class ContextualGaussianProcessUpperConfidenceBoundWithRelevanceLearning:
    """ Implementation of CGP-UCB-RL """

    def __init__(self, horizon, dx, da, arm_limits=(0, 1.00000001), arm_granularity=0.01, noise_sigma=1,
                 confidence_scale=1.0, fit_ard_period=1):
        pass
        self.horizon, self.dx, self.da = horizon, dx, da
        self.t, self.played_points_hist, self.rews_hist = 0, np.zeros(shape=(horizon, dx + da)), np.zeros(shape=(horizon,))
        arm_grid_1d = np.arange(arm_limits[0], arm_limits[1], arm_granularity)
        self.arm_grid = np.array(list(itertools.product(arm_grid_1d, repeat=da)))
        self.kernel_fn = rbf_kernel_w_linear_coefficients
        self.noise_sigma = noise_sigma
        self.cs = confidence_scale
        self.last_played_point = None
        self.fit_ard_period = fit_ard_period
        self.best_ard_params = np.ones(shape=(dx + da, ))

    def select_arm(self, context):
        if self.t == 0:
            rand_int = np.random.randint(low=0, high=self.arm_grid.shape[0])
            selected_arm = self.arm_grid[rand_int]
            self.last_played_point = np.concatenate((context, selected_arm))
        else:
            context_arm_grid = self.calculate_context_arm_grid(context=context)
            if self.t % self.fit_ard_period == 0 and self.t != 0:
                self.best_ard_params, _ = self.calculate_discrete_best_ard_method_unknown_rel_dims()
            mean_est, var_est = self.calculate_posterior_mean_var(context_arm_grid=context_arm_grid)
            cgp_arm_ind, cgp_high_ucb = self.calculate_highest_ucb_index(1.0, mean_est, var_est)
            # TODO: add highest variance arm selection
            self.last_played_point = context_arm_grid[cgp_arm_ind]
            selected_arm = context_arm_grid[cgp_arm_ind, self.dx:]
        return selected_arm

    def update_statistics(self, reward):
        self.played_points_hist[self.t] = self.last_played_point
        self.rews_hist[self.t] = reward
        self.t += 1

    def calculate_context_arm_grid(self, context):
        """For each arm, calculates the points when concatenated with the context"""
        contexts_repeated = np.repeat(np.expand_dims(context, axis=0), repeats=self.arm_grid.shape[0], axis=0)
        context_arm_grid = np.concatenate((contexts_repeated, self.arm_grid), axis=-1)
        return context_arm_grid

    def calculate_posterior_mean_var(self, context_arm_grid):
        """Calculates statistics of the posterior distribution of expected reward function, possibly ignoring some dimensions acc. to
        ard_coeffs """
        # all points played so far
        data = self.played_points_hist[:self.t]
        # kernels between all possible context-arms and the previous rounds
        kernel_vectors = self.kernel_fn(context_arm_grid, data, self.best_ard_params)
        # kernel matrix of data
        kernel_matrix = self.kernel_fn(data, data, self.best_ard_params)
        c_matrix = kernel_matrix + (self.noise_sigma ** 2) * np.eye(data.shape[0])
        c_matrix_inv = np.linalg.inv(c_matrix)
        mu_ests_vector = np.matmul(kernel_vectors, np.matmul(c_matrix_inv, self.rews_hist[:self.t]))  # mean estimation
        sigma_ests_first_term = np.diag(self.kernel_fn(context_arm_grid, context_arm_grid, self.best_ard_params))
        sigma_ests_second_term = np.diag(np.matmul(kernel_vectors, np.matmul(c_matrix_inv, kernel_vectors.T)))
        sigma_ests_vector = sigma_ests_first_term - sigma_ests_second_term  # variance estimation
        return mu_ests_vector, sigma_ests_vector

    def calculate_negative_log_likelihood(self):
        """Calculates the negative log marginal likelihood, possibly ignoring some dimensions acc. to ard_coeffs"""
        data = self.played_points_hist[:self.t]
        kernel_matrix = self.kernel_fn(data, data, self.best_ard_params)
        c_matrix = kernel_matrix + (self.noise_sigma ** 2) * np.eye(data.shape[0])
        c_matrix_inv = np.linalg.inv(c_matrix)
        first_term = np.matmul(self.rews_hist[:self.t].T, np.matmul(c_matrix_inv, self.rews_hist[:self.t]))
        second_term = np.log(np.linalg.det(c_matrix))
        return first_term + second_term

    def calculate_discrete_best_ard_method_unknown_rel_dims(self):
        """Optimization of NLL, find which dimensions should be ignored"""
        dx_powset, da_powset = calculate_power_set(np.arange(0, self.dx)), calculate_power_set(np.arange(0, self.da))
        dx_powset, da_powset = dx_powset[1:], da_powset[1:]  # calculate power set dimensions for contexts and arms and remove empty sets
        nlls, ard_params_list = [], []
        # for each combination of context and arm dimension tuples calculate NLL
        for dx_r in dx_powset:
            dx_r_np = np.array(dx_r)
            context_ard_params = np.zeros(shape=(self.dx,))
            context_ard_params[dx_r_np] = 1
            for da_r in da_powset:
                da_r_np = np.array(da_r)
                arm_ard_params = np.zeros(shape=(self.da,))
                arm_ard_params[da_r_np] = 1
                ard_params = np.concatenate((context_ard_params, arm_ard_params))
                ard_params = np.squeeze(ard_params)
                nll = self.calculate_negative_log_likelihood()
                nlls.append(nll)
                ard_params_list.append(ard_params)
        nlls = np.array(nlls)
        ard_params_list = np.array(ard_params_list)
        argmin_ind = np.argmin(nlls)  # find best NLL index
        best_ard_params = ard_params_list[argmin_ind]  # select coefficient that best ignore the dimensions
        return best_ard_params, nlls[argmin_ind]

    def calculate_highest_ucb_index(self, beta, means, stds, return_multiple=False):
        """ calculate highest UCB """
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