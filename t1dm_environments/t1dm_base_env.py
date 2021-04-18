import numpy as np
from imblearn.over_sampling import RandomOverSampler


class T1DMBaseEnvironment:

    def __init__(self, patients_data):
        """ Sets up the T1DM environment using given data dictionary for multiple patients """
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

            self.patient_ids.extend(i * np.ones(shape=(num_data_for_patient,), dtype=np.int))
            self.patient_occurrences.append(num_data_for_patient)

        self.patient_ids = np.array(self.patient_ids)
        self.patient_weights = np.array(self.patient_occurrences) / np.sum(self.patient_occurrences)
        self.contexts, self.arms, self.reward_variables = np.array(contexts), np.array(arms), np.array(reward_variables)

        # save a copy of the original values from the dataset before normalization/standardization and resampling
        self.contexts_og = np.copy(self.contexts)
        self.arms_og = np.copy(self.arms)
        self.reward_variables_og = np.copy(self.reward_variables)

        # other fields
        self.num_patients = num_patients
        self.patient_context_density_estimators = []
        self.context_range = (-np.inf, np.inf)
        self.arm_range = (-np.inf, np.inf)
        self.regressor = None

    def fit_patient_context_density_estimators(self, density_estimator):
        for i in range(self.num_patients):
            pat_i = self.contexts[self.patient_ids == i]
            self.patient_context_density_estimators.append(density_estimator.fit(pat_i))

    def random_over_sample_patients(self):
        ros = RandomOverSampler()
        ros.fit_resample(np.zeros(shape=(self.patient_ids.shape[0], 1)), self.patient_ids)
        ros_inds = ros.sample_indices_
        self.contexts, self.arms, self.reward_variables = self.contexts[ros_inds], self.arms[ros_inds], self.reward_variables[ros_inds]

    def get_original_dataset(self):
        """ Returns the original values from the dataset formatted suitable for CMAB applications """
        return self.contexts_og, self.arms_og, self.reward_variables_og, self.patient_ids

    def get_context(self):
        """ Generate a context vector """
        # get index for a patient randomly
        patient_select = int(np.random.choice(a=self.num_patients, p=self.patient_weights))
        # sample context from that patients density estimator
        context_sample = self.patient_context_density_estimators[patient_select].sample(n_samples=1)[0]
        # check if context is between self.context_range
        range_check = (self.context_range[0] <= context_sample) * (context_sample <= self.context_range[1])
        # if not sample until context is in self.context_range for all dimensions
        while np.sum(range_check) < self.dx:
            context_sample = self.patient_context_density_estimators[patient_select].sample(n_samples=1)[0]
            range_check = (self.context_range[0] <= context_sample) * (context_sample <= self.context_range[1])
        # return the context and patient id from which the context is sampled
        return np.squeeze(context_sample), np.squeeze(patient_select)

    def get_reward_at(self, context, arm, noise_std=0.0):
        """ return the reward and the generated cgm according to the context and the selected arm """
        # concatenate contexts and arms
        p = np.concatenate((context, arm))[np.newaxis, :]
        # create noise value if desired
        noise = np.random.normal(loc=0.0, scale=noise_std, size=1)
        # generate a cgm value using the regressor model and the generated noise
        generated_cgm = np.squeeze(self.regressor.predict(p)) + noise
        # as a cgm value cannot be negative, clip the cgm value if below zero
        generated_cgm = 0.0 if generated_cgm < 0.0 else generated_cgm
        # return both the reward and the cgm value
        reward = np.squeeze(self.cgm_to_reward(generated_cgm))
        return reward, generated_cgm

    @staticmethod
    def cgm_to_reward(cgm):
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
