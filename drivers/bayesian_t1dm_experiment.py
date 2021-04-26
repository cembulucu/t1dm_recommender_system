import pickle
import time

import numpy as np

from bandit_algorithms.cgp_ucb import ContextualGaussianProcessUpperConfidenceBoundAlgorithm
from bandit_algorithms.cgp_ucb_rl import ContextualGaussianProcessUpperConfidenceBoundWithRelevanceLearning
from bandit_algorithms.uniform_random_bandit import UniformRandomBandit
from t1dm_environments.t1dm_gp_env import T1DMGaussianProcessEnvironment

if __name__ == '__main__':
    # define experiment setup
    n_repeats = 20
    horizon, verbose_period = 100, 20

    # load data
    root_path = '../t1dm_pickles/'
    with open(root_path + "patients_data.pkl", "rb") as f:
        patients_data = pickle.load(f)

    # initialize what to save during experiments: which patient generated the context, rewards and cgms for each algorithm
    patient_info_per_round = np.zeros(shape=(n_repeats, horizon))

    rew_hist_cgpucb, cgm_hist_cgpucb = np.zeros(shape=(n_repeats, horizon)), np.zeros(shape=(n_repeats, horizon))
    rew_hist_cgpucbrl, cgm_hist_cgpucbrl = np.zeros(shape=(n_repeats, horizon)), np.zeros(shape=(n_repeats, horizon))
    rew_hist_urb, cgm_hist_urb = np.zeros(shape=(n_repeats, horizon)), np.zeros(shape=(n_repeats, horizon))

    # perform the experiment independently 'n_repeats' times
    for r in range(n_repeats):
        # create the T1DM environment
        bandit_env = T1DMGaussianProcessEnvironment(patients_data)
        dx, da = bandit_env.dx, bandit_env.da

        cgpucb = ContextualGaussianProcessUpperConfidenceBoundAlgorithm(horizon=horizon, dx=dx, da=da, space_diameter=34, delta=0.01,
                                                                        confidence_scale=0.01)
        cgpucbrl = ContextualGaussianProcessUpperConfidenceBoundWithRelevanceLearning(horizon=horizon, dx=dx, da=da,
                                                                                      confidence_scale=0.01, fit_ard_period=5)
        urb = UniformRandomBandit(da=da)

        start_clock = time.time()
        # at each time step in until the horizon
        for t in range(horizon):
            # generate a context from the env
            c, patient_id = bandit_env.get_context()

            # determine played arms for each algorithm
            y_cgpucb = cgpucb.select_arm(c)
            y_cgpucbrl = cgpucbrl.select_arm(c)
            y_urb = urb.select_arm()

            # get reward for each algorithm
            r_cgpucb, cgm_cgpucb = bandit_env.get_reward_at(c, y_cgpucb, noise_std=5.0)
            r_cgpucbrl, cgm_cgpucbrl = bandit_env.get_reward_at(c, y_cgpucbrl, noise_std=5.0)
            r_urb, cgm_urb = bandit_env.get_reward_at(c, y_urb, noise_std=5.0)

            # update each algorithm with the reward
            cgpucb.update_statistics(r_cgpucb)
            cgpucbrl.update_statistics(r_cgpucbrl)

            # save statistics regarding rewards and such
            rew_hist_cgpucb[r, t], cgm_hist_cgpucb[r, t] = r_cgpucb, cgm_cgpucb
            rew_hist_cgpucbrl[r, t], cgm_hist_cgpucbrl[r, t] = r_cgpucbrl, cgm_cgpucbrl
            rew_hist_urb[r, t], cgm_hist_urb[r, t] = r_urb, cgm_urb
            patient_info_per_round[r, t] = patient_id

            # if verbose_period is reached print some info regarding how the experiment is going
            if (t % verbose_period == 0 and t != 0) or t == horizon-1:
                print('')
                end_lap = time.time()
                print('Repeat: ', r, ', round: ', t, ', total time elapsed: %.2f' % (end_lap - start_clock))

                rews_cgpucb = np.sum(rew_hist_cgpucb[r, :t])
                rews_cgpucbrl = np.sum(rew_hist_cgpucbrl[r, :t])

                print(np.round(np.histogram(cgm_hist_cgpucb[r, :t], bins=[0, 80, 180, np.inf])[0] / t, decimals=2))
                print(np.round(np.histogram(cgm_hist_cgpucbrl[r, :t], bins=[0, 80, 180, np.inf])[0] / t, decimals=2))
                print(np.round(np.histogram(cgm_hist_urb[r, :t], bins=[0, 80, 180, np.inf])[0] / t, decimals=2))
