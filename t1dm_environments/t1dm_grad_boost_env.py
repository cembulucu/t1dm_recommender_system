import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler


def partial_reward_fn(cgm):
    """ reward function that translates cgm values into rewards in [0,1] range """
    hypoglycemia_limit, optimal_cgm_lower_limit, optimal_cgm_upper_limit, hyperglycemia_limit = 80, 90, 130, 180
    if hypoglycemia_limit < cgm <= optimal_cgm_lower_limit:
        return (cgm - hypoglycemia_limit) / (optimal_cgm_lower_limit - hypoglycemia_limit)
    elif optimal_cgm_lower_limit < cgm <= optimal_cgm_upper_limit:
        return 1.0
    elif optimal_cgm_upper_limit < cgm <= hyperglycemia_limit:
        return (hyperglycemia_limit - cgm) / (hyperglycemia_limit - optimal_cgm_upper_limit)
    else:
        return 0.0


class T1DMEnvironment:
    """ """
    def __init__(self, patients_data):
        """ Sets up the T1DM environment using given data dictionary for multiple patients """
        self.rew_fn = partial_reward_fn
        context_names = ['prev_cgms', 'skin_temps', 'air_temps', 'gsrs', 'steps', 'exercises', 'heart_rates', 'basals', 'meals']
        arm_names = ['boluses']
        reward_variable_name = ['next_cgms']
        num_patients = len(patients_data)

        self.dx, self.da = len(context_names), len(arm_names)
        self.patient_ids = []
        self.patient_occurrences = []
        contexts, arms, reward_variables = [], [], []
        # for each patient extract contexts, arms and reward variables
        for i in range(num_patients):
            pat_i_data_dict = patients_data[i]
            num_data_for_patient = pat_i_data_dict[reward_variable_name[0]].shape[0]
            contexts.extend(np.array([pat_i_data_dict[cn] for cn in context_names]).T)
            arms.extend(np.array([pat_i_data_dict[an] for an in arm_names]).T)
            reward_variables.extend(np.array([pat_i_data_dict[rn] for rn in reward_variable_name]).T)

            self.patient_ids.extend(i * np.ones(shape=(num_data_for_patient, ), dtype=np.int))
            self.patient_occurrences.append(num_data_for_patient)

        self.patient_ids = np.array(self.patient_ids)
        self.patient_weights = np.array(self.patient_occurrences) / np.sum(self.patient_occurrences)
        contexts, arms, reward_variables = np.array(contexts), np.array(arms), np.array(reward_variables)

        # save a copy of the original values from the dataset before normalization and resampling
        self.contexts_og = np.copy(contexts)
        self.arms_og = np.copy(arms)
        self.reward_variables = np.copy(reward_variables)

        # normalize the contexts and arms so that all values reside withing [0, 1] range
        contexts_scaler, arms_scaler = MinMaxScaler(), MinMaxScaler()
        contexts, arms = contexts_scaler.fit_transform(contexts), arms_scaler.fit_transform(arms)

        # for each patient construct Gaussians, these models are going to be used for context generation
        self.patient_context_density_estimators = []
        for i in range(num_patients):
            pat_i = contexts[self.patient_ids == i]
            self.patient_context_density_estimators.append(GaussianMixture(n_init=10, n_components=1).fit(pat_i))

        # oversample the contexts, arms and reward variables so all number of data for each patient is equal for the regression model
        ros = RandomOverSampler()
        ros.fit_resample(contexts, self.patient_ids)
        rus_inds = ros.sample_indices_
        contexts, arms, reward_variables = contexts[rus_inds], arms[rus_inds], reward_variables[rus_inds]

        # Use gradient boosting regressor to model how patients with different states(contexts) respond to different bolus doses(arms),
        # where the output is the resulting cgm values
        feats_data = np.concatenate((contexts, arms), axis=-1)
        regressor = GradientBoostingRegressor(n_estimators=100, max_depth=5, loss='huber')
        regressor.fit(feats_data, np.squeeze(reward_variables))

        self.feats_data = np.copy(feats_data)
        self.contexts_scaler = contexts_scaler
        self.arms_scaler = arms_scaler
        self.regressor_model = regressor

    def get_original_dataset(self):
        """ Returns the original values from the dataset formatted suitable for CMAB applications """
        return self.contexts_og, self.arms_og, self.reward_variables, self.patient_ids

    def get_context(self):
        """ Generate a context vector """
        # get index for a patient randomly
        patient_select = int(np.random.choice(a=[0, 1, 2, 3, 4, 5], p=self.patient_weights))
        # sample context from that patients Gaussian
        context_sample = self.patient_context_density_estimators[patient_select].sample(n_samples=1)[0]
        # check if context is between 0 and 1
        range_check = (0 <= context_sample) * (context_sample <= 1)
        # if not sample until context is in range [0, 1] for all dimensions, essentially create a truncated gaussian
        while not range_check.all():
            context_sample = self.patient_context_density_estimators[patient_select].sample(n_samples=1)[0]
            range_check = (0 <= context_sample) * (context_sample <= 1)
        # return the context and patient id from which the context is sampled
        return np.squeeze(context_sample), np.squeeze(patient_select)

    def get_reward_at(self, context, arm, use_noise=True):
        """ return the reward and the generated cgm according to the context and the selected arm """
        # concatenate contexts and arms
        p = np.concatenate((context, arm))[np.newaxis, :]
        # create noise value if desired
        noise = np.random.normal(loc=0.0, scale=5.0, size=1) if use_noise else 0.0
        # generate a cgm value using the regressor model and the generated noise
        generated_cgm = np.squeeze(self.regressor_model.predict(p)) + noise
        # as a cgm value cannot be negative, clip the cgm value if below zero
        if generated_cgm < 0:
            generated_cgm = 0
        # return both the reward and the cgm value
        return np.squeeze(self.rew_fn(generated_cgm)), generated_cgm
