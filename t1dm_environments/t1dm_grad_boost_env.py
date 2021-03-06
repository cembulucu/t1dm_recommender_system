import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from t1dm_environments.t1dm_base_env import T1DMBaseEnvironment


class T1DMGradientBoostingEnvironment(T1DMBaseEnvironment):
    """ T1DM Environment that models blood glucose behaviour using Gradient Boosting """
    def __init__(self, patients_data):
        super().__init__(patients_data)

        # normalize the contexts and arms so that all values reside withing [0, 1] range
        self.context_range, self.arm_range = (0, 1), (0, 1)
        contexts_scaler, arms_scaler = MinMaxScaler(feature_range=self.context_range), MinMaxScaler(feature_range=self.arm_range)
        self.contexts, self.arms = contexts_scaler.fit_transform(self.contexts), arms_scaler.fit_transform(self.arms)

        # for each patient construct Gaussians, these models are going to be used for context generation
        self.fit_patient_context_density_estimators(density_estimator=GaussianMixture(n_init=10, n_components=1))

        # oversample the contexts, arms and reward variables so all number of data for each patient is equal for the regression model
        self.random_over_sample_patients()

        # Use gradient boosting regressor to model how patients with different states(contexts) respond to different bolus doses(arms),
        # where the output is the resulting cgm values
        feats_data = np.concatenate((self.contexts, self.arms), axis=-1)
        self.regressor = GradientBoostingRegressor(n_estimators=100, max_depth=5, loss='huber')
        self.regressor.fit(feats_data, np.squeeze(self.reward_variables))


