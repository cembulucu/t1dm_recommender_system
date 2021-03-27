import pickle
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.stats as spstats

from bandit_algorithms.c_hoo import ContextualHierarchicalOptimisticOptimizationBandit as CHoo
from bandit_algorithms.cmab_rl import ContextualMultiArmedBanditWithRelevanceLearning as CmabRl
from bandit_algorithms.iup import InstanceBasedUniformPartitioningBandit as Iup
from bandit_algorithms.uniform_random_bandit import UniformRandomBandit as Urb
from t1dm_environments.t1dm_grad_boost_env import T1DMEnvironment

if __name__ == '__main__':
    # define experiment setup
    n_repeats = 20
    npz_str = 'CMABRL_vs_IUP_vs_CHOO_t1dm_reps' + str(n_repeats)
    horizon, verbose_period = 100000, 1000

    # load data
    root_path = '../t1dm_pickles/'
    with open(root_path + "patients_data.pkl", "rb") as f:
        patients_data = pickle.load(f)

    # initialize what to save during experiments: which patient generated the context, rewards and cgms for each algorithm
    patient_info_per_round = np.zeros(shape=(n_repeats, horizon))
    rew_hist_cmabrl, cgm_hist_cmabrl = np.zeros(shape=(n_repeats, horizon)), np.zeros(shape=(n_repeats, horizon))
    rew_hist_choo, cgm_hist_choo = np.zeros(shape=(n_repeats, horizon)), np.zeros(shape=(n_repeats, horizon))
    rew_hist_iup, cgm_hist_iup = np.zeros(shape=(n_repeats, horizon)), np.zeros(shape=(n_repeats, horizon))
    rew_hist_urb, cgm_hist_urb = np.zeros(shape=(n_repeats, horizon)), np.zeros(shape=(n_repeats, horizon))

    # perform the experiment independently 'n_repeats' times
    for r in range(n_repeats):
        # create the T1DM environment
        bandit_env = T1DMEnvironment(patients_data)
        dx, da = bandit_env.dx, bandit_env.da

        # initialize bandit algorithms
        cmabrl = CmabRl(horizon=horizon, dx=dx, da=da, dx_bar=1, da_bar=1, lip_c=1.0, conf_scale=0.001)
        choo = CHoo(horizon=horizon, dx=dx, da=da, conf_scale=0.1)
        iup = Iup(horizon=horizon, dx=dx, da=da, conf_scale=0.05)
        urb = Urb(da=da)

        start_clock = time.time()
        # at each time step in until the horizon
        for t in range(horizon):
            # generate a context from the env
            c, patient_id = bandit_env.get_context()

            # determine played arms for each algorithm
            y_cmabrl = cmabrl.determine_arm_one_round(c)
            y_choo = choo.determine_arm_one_round(c)
            y_iup = iup.determine_arm_one_round(c)
            y_urb = urb.determine_arm_one_round(c)

            # get reward for each algorithm
            r_cmabrl, cgm_cmabrl = bandit_env.get_reward_at(c, y_cmabrl, use_noise=True)
            r_choo, cgm_choo = bandit_env.get_reward_at(c, y_choo, use_noise=True)
            r_iup, cgm_iup = bandit_env.get_reward_at(c, y_iup, use_noise=True)
            r_urb, cgm_urb = bandit_env.get_reward_at(c, y_urb, use_noise=True)

            # update each algorithm with the reward
            cmabrl.update_statistics(r_cmabrl)
            choo.update_statistics(r_choo)
            iup.update_statistics(r_iup)
            urb.update_statistics(r_urb)

            # save statistics regarding rewards and such
            rew_hist_cmabrl[r, t], cgm_hist_cmabrl[r, t] = r_cmabrl, cgm_cmabrl
            rew_hist_choo[r, t], cgm_hist_choo[r, t] = r_choo, cgm_choo
            rew_hist_iup[r, t], cgm_hist_iup[r, t] = r_iup, cgm_iup
            rew_hist_urb[r, t], cgm_hist_urb[r, t] = r_urb, cgm_urb
            patient_info_per_round[r, t] = patient_id

            # if verbose_period is reached print some infor regarding how the experiment is going
            if t % verbose_period == 0 and t != 0:
                print('')
                end_lap = time.time()
                print('Repeat: ', r, ', round: ', t, ', total time elapsed: %.2f' % (end_lap - start_clock))

                rews_cmabrl = np.sum(rew_hist_cmabrl[r])
                rews_choo = np.sum(rew_hist_choo[r])
                rews_iup = np.sum(rew_hist_iup[r])
                rews_urb = np.sum(rew_hist_urb[r])

                print('CMAB-RL stats, avg rew: %.3f' % (rews_cmabrl / (t + 1)), ', sum rew: %.0f' % rews_cmabrl)
                print('C-HOO stats, avg rew: %.3f' % (rews_choo / (t + 1)), ', sum rew: %.0f' % rews_choo)
                print('IUP stats, avg rew: %.3f' % (rews_iup / (t + 1)), ', sum rew: %.0f' % rews_iup)
                print('URB stats, avg rew: %.3f' % (rews_urb / (t + 1)), ', sum rew: %.0f' % rews_urb)









