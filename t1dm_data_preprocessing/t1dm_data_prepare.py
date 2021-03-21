import pickle
import xml.etree.ElementTree as eT
import numpy as np
from datetime import datetime, timedelta


def str_to_datetime(date_str):
    """ converts a string date time into a datetime object, example date: date_str = "08-12-2021 22:14:00" """
    ymd, hms = date_str.split()
    day, month, year = ymd.split('-')
    hour, minute, seconds = hms.split(':')
    date_time = datetime(int(year), int(month), int(day), int(hour), int(minute), int(seconds))
    return date_time


def xml_read_variables(path):
    """reads each variable and saves it in a dictionary """
    tree = eT.parse(path)  # xml parser
    root = tree.getroot()

    meal_values, cgm_values, skin_temp_values, air_temp_values, bolus_values = None, None, None, None, None
    heart_rate_values, gsr_values, step_values, exercise_values, basal_values, temp_basal_values = None, None, None, None, None, None

    meal_dates, cgm_dates, skin_temp_dates, air_temp_dates, bolus_dates = None, None, None, None, None
    heart_rate_dates, gsr_dates, step_dates, exercise_dates, basal_dates = None, None, None, None, None
    temp_basal_begin_dates, temp_basal_end_dates = None, None

    # the for loops below extract each event from the corresponding event type
    for meal in root.iter('meal'):
        meal_values = np.array([float(event.get('carbs')) for event in meal.iter('event')])
        meal_dates = np.array([str_to_datetime(event.get('ts')) for event in meal.iter('event')])

    for cgm in root.iter('glucose_level'):
        cgm_values = np.array([float(event.get('value')) for event in cgm.iter('event')])
        cgm_dates = np.array([str_to_datetime(event.get('ts')) for event in cgm.iter('event')])

    for skin_temp in root.iter('basis_skin_temperature'):
        skin_temp_values = np.array([float(event.get('value')) for event in skin_temp.iter('event')])
        skin_temp_dates = np.array([str_to_datetime(event.get('ts')) for event in skin_temp.iter('event')])

    for air_temp in root.iter('basis_air_temperature'):
        air_temp_values = np.array([float(event.get('value')) for event in air_temp.iter('event')])
        air_temp_dates = np.array([str_to_datetime(event.get('ts')) for event in air_temp.iter('event')])

    for bolus in root.iter('bolus'):
        bolus_values = np.array([float(event.get('dose')) for event in bolus.iter('event')])
        bolus_dates = np.array([str_to_datetime(event.get('ts_begin')) for event in bolus.iter('event')])

    for heart_rate in root.iter('basis_heart_rate'):
        heart_rate_values = np.array([float(event.get('value')) for event in heart_rate.iter('event')])
        heart_rate_dates = np.array([str_to_datetime(event.get('ts')) for event in heart_rate.iter('event')])

    for gsr in root.iter('basis_gsr'):
        gsr_values = np.array([float(event.get('value')) for event in gsr.iter('event')])
        gsr_dates = np.array([str_to_datetime(event.get('ts')) for event in gsr.iter('event')])

    for step in root.iter('basis_steps'):
        step_values = np.array([float(event.get('value')) for event in step.iter('event')])
        step_dates = np.array([str_to_datetime(event.get('ts')) for event in step.iter('event')])

    for exercise in root.iter('exercise'):
        exercise_values = np.array([float(event.get('intensity')) * float(event.get('duration')) for event in exercise.iter('event')])
        exercise_dates = np.array([str_to_datetime(event.get('ts')) for event in exercise.iter('event')])

    for basal in root.iter('basal'):
        basal_values = np.array([float(event.get('value')) for event in basal.iter('event')])
        basal_dates = np.array([str_to_datetime(event.get('ts')) for event in basal.iter('event')])

    for temp_basal in root.iter('temp_basal'):
        temp_basal_values = np.array([float(event.get('value')) for event in temp_basal.iter('event')])
        temp_basal_begin_dates = np.array([str_to_datetime(event.get('ts_begin')) for event in temp_basal.iter('event')])
        temp_basal_end_dates = np.array([str_to_datetime(event.get('ts_end')) for event in temp_basal.iter('event')])

    """change basal and temp basal so that temp_basal overrides basal and they become a single variable
    (when temp_basal is active in a time period, regular basal values are overwritten)
    the basal values are in about 4 hour intervals, to make it easier we will calculate the basal value at each cgm measurement """
    # see where each cgm index needs to be placed in basal_dates, st. array remains sorted
    inds = np.searchsorted(basal_dates, cgm_dates) - 1

    # simply oversample the basal rates and update basal_dates into 5 min intervals
    basal_values, basal_dates = basal_values[inds], cgm_dates

    # see where temp_basals are active
    inds_for_begin = np.searchsorted(basal_dates, temp_basal_begin_dates)
    inds_for_end = np.searchsorted(basal_dates, temp_basal_end_dates)

    for k, (i, j) in enumerate(zip(inds_for_begin, inds_for_end)):
        # replace the value of basal with temp_basal where temp_basal is active
        basal_values[i: j] = temp_basal_values[k]

    data_dict = {'bolus_values': bolus_values, 'bolus_dates': bolus_dates,
                 'basal_values': basal_values, 'basal_dates': basal_dates,
                 'meal_values': meal_values, 'meal_dates': meal_dates,
                 'cgm_values': cgm_values, 'cgm_dates': cgm_dates,
                 'skin_temp_values': skin_temp_values, 'skin_temp_dates': skin_temp_dates,
                 'air_temp_values': air_temp_values, 'air_temp_dates': air_temp_dates,
                 'gsr_values': gsr_values, 'gsr_dates': gsr_dates,
                 'step_values': step_values, 'step_dates': step_dates,
                 'exercise_values': exercise_values, 'exercise_dates': exercise_dates,
                 'heart_rate_values': heart_rate_values, 'heart_rate_dates': heart_rate_dates}

    return data_dict


def prepare_dateset(data_dict, prev_time_interval, reward_cgm_offset, reward_cgm_interval_length):
    """ this method prepares the bandit like dataset, at each round we will have a single value for each variable """
    boluses, basals, meals, prev_cgms, next_cgms_mean = [], [], [], [], []
    skin_temps, air_temps, gsrs, steps, exercises, heart_rates = [], [], [], [], [], []
    for i, (bolus_val, bolus_date) in enumerate(zip(data_dict['bolus_values'], data_dict['bolus_dates'])):
        # for each bolus event, check previous events for contexts and upcoming events for rewards
        prev_hour = bolus_date - timedelta(minutes=prev_time_interval)

        # get cgms (skip if not available unlike other variables,
        # this is both relevant and replacing with 0 is not correct, also it is related to reward)
        cgm_values, cgm_dates = data_dict['cgm_values'], data_dict['cgm_dates']
        prev_cgm_values = cgm_values[(prev_hour <= cgm_dates) & (cgm_dates <= bolus_date)]
        next_cgm_values = cgm_values[(bolus_date + timedelta(minutes=reward_cgm_offset) <= cgm_dates) &
                                     (cgm_dates <= bolus_date + timedelta(minutes=reward_cgm_offset + reward_cgm_interval_length))]
        if prev_cgm_values.shape[0] < 1 or next_cgm_values.shape[0] < 1:
            continue
        prev_cgm_avg, next_cgm_avg, next_cgms_up, next_cgms_down = np.mean(prev_cgm_values), np.mean(next_cgm_values), np.max(
            next_cgm_values), np.min(next_cgm_values)

        # get meals(if unavailable means no meal)
        meal_values, meal_dates = data_dict['meal_values'], data_dict['meal_dates']
        prev_meal_values = meal_values[(prev_hour <= meal_dates) & (meal_dates <= bolus_date)]
        if prev_meal_values.shape[0] < 1:
            prev_meal_values = 0
        prev_meal_avg = np.sum(prev_meal_values)

        # get basals(if unavailable assume no basal injection)
        basal_values, basal_dates = data_dict['basal_values'], data_dict['basal_dates']
        prev_basal_values = basal_values[(prev_hour <= basal_dates) & (basal_dates <= bolus_date)]
        if prev_basal_values.shape[0] < 1:
            prev_basal_values = 0
        prev_basal_avg = np.mean(prev_basal_values)

        # get skin_temp(if unavailable replace with the mean over all instances)
        skin_temp_values, skin_temp_dates = data_dict['skin_temp_values'], data_dict['skin_temp_dates']
        prev_skin_temp_values = skin_temp_values[(prev_hour <= skin_temp_dates) & (skin_temp_dates <= bolus_date)]
        if prev_skin_temp_values.shape[0] < 1:
            prev_skin_temp_values = np.mean(skin_temp_values)
        prev_skin_temp_avg = np.mean(prev_skin_temp_values)

        # get air_temp(if unavailable replace with the mean over all instances)
        air_temp_values, air_temp_dates = data_dict['air_temp_values'], data_dict['air_temp_dates']
        prev_air_temp_values = air_temp_values[(prev_hour <= air_temp_dates) & (air_temp_dates <= bolus_date)]
        if prev_air_temp_values.shape[0] < 1:
            prev_air_temp_values = np.mean(air_temp_values)
        prev_air_temp_avg = np.mean(prev_air_temp_values)

        # get gsr(if unavailable replace with the mean over all instances)
        gsr_values, gsr_dates = data_dict['gsr_values'], data_dict['gsr_dates']
        prev_gsr_values = gsr_values[(prev_hour <= gsr_dates) & (gsr_dates <= bolus_date)]
        if prev_gsr_values.shape[0] < 1:
            prev_gsr_values = np.mean(gsr_values)
        prev_gsr_avg = np.mean(prev_gsr_values)

        # get step(if unavailable assume no steps)
        step_values, step_dates = data_dict['step_values'], data_dict['step_dates']
        prev_step_values = step_values[(prev_hour <= step_dates) & (step_dates <= bolus_date)]
        if prev_step_values.shape[0] < 1:
            prev_step_values = 0
        prev_step_avg = np.sum(prev_step_values)

        # get exercise(if unavailable assume no exercise)
        exercise_values, exercise_dates = data_dict['exercise_values'], data_dict['exercise_dates']
        prev_exercise_values = exercise_values[(prev_hour <= exercise_dates) & (exercise_dates <= bolus_date)]
        if prev_exercise_values.shape[0] < 1:
            prev_exercise_values = 0
        prev_exercise_avg = np.sum(prev_exercise_values)

        # get heart_rate(if unavailable replace with the mean over all instances)
        heart_rate_values, heart_rate_dates = data_dict['heart_rate_values'], data_dict['heart_rate_dates']
        prev_heart_rate_values = heart_rate_values[(prev_hour <= heart_rate_dates) & (heart_rate_dates <= bolus_date)]
        if prev_heart_rate_values.shape[0] < 1:
            prev_heart_rate_values = np.mean(heart_rate_values)
        prev_heart_rate_avg = np.mean(prev_heart_rate_values)

        # save variables
        boluses.append(bolus_val)
        prev_cgms.append(prev_cgm_avg)
        next_cgms_mean.append(next_cgm_avg)
        meals.append(prev_meal_avg)
        basals.append(prev_basal_avg)
        skin_temps.append(prev_skin_temp_avg)
        air_temps.append(prev_air_temp_avg)
        gsrs.append(prev_gsr_avg)
        steps.append(prev_step_avg)
        exercises.append(prev_exercise_avg)
        heart_rates.append(prev_heart_rate_avg)

    data_dict = {'boluses': np.array(boluses), 'prev_cgms': np.array(prev_cgms), 'next_cgms_mean': np.array(next_cgms_mean),
                 'meals': np.array(meals), 'basals': np.array(basals), 'skin_temps': np.array(skin_temps),
                 'air_temps': np.array(air_temps), 'gsrs': np.array(gsrs), 'steps': np.array(steps),
                 'exercises': np.array(exercises), 'heart_rates': np.array(heart_rates)}

    return data_dict


if __name__ == '__main__':
    save_output = True
    # extract and save data
    xml_root_path = '../t1dm_xmls/'
    xml_paths = ['patient_559_merged.xml', 'patient_563_merged.xml', 'patient_570_merged.xml',
                 'patient_575_merged.xml', 'patient_588_merged.xml', 'patient_591_merged.xml']

    xml_paths = [xml_root_path + p for p in xml_paths]
    patient_data = []
    for xml_path in xml_paths:
        patient_data_dict = xml_read_variables(xml_path)
        patient_data.append(prepare_dateset(patient_data_dict, prev_time_interval=30, reward_cgm_offset=30, reward_cgm_interval_length=90))
        pass

    if save_output:
        with open("../t1dm_pickles/patients_data.pkl", 'wb') as f:
            pickle.dump(patient_data, f)
