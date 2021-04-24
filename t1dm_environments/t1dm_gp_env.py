import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from t1dm_environments.t1dm_base_env import T1DMBaseEnvironment


class T1DMGaussianProcessEnvironment(T1DMBaseEnvironment):
    """ T1DM Environment that models blood glucose behaviour using Gaussian Process """
    def __init__(self, patients_data):
        super().__init__(patients_data)

        # normalize the contexts and arms so that all features have zero mean and unit variance
        contexts_scaler, arms_scaler = StandardScaler(), StandardScaler()
        self.contexts, self.arms = contexts_scaler.fit_transform(self.contexts), arms_scaler.fit_transform(self.arms)

        # for each patient construct Gaussians, these models are going to be used for context generation
        self.fit_patient_context_density_estimators(density_estimator=GaussianMixture(n_init=10, n_components=1))

        # oversample the contexts, arms and reward variables so all number of data for each patient is equal for the regression model
        self.random_over_sample_patients()

        # fit Gaussian process regressor that allows different weights for different dimensions
        feats_data = np.concatenate((self.contexts, self.arms), axis=-1)
        kernel = RBF(length_scale=[1] * 10, length_scale_bounds=(1, 100))
        self.regressor = GaussianProcessRegressor(kernel=kernel, alpha=1,)
        self.regressor.fit(feats_data, np.squeeze(self.reward_variables))
