import pandas as pd
import numpy as np
from influxdb import DataFrameClient

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

from functions import FUNCS


def get_data():
    client = DataFrameClient('192.168.1.103', 8086, 'admin', 'admin', 'feuchtigkeit')
    data = client.query(
        'select "object_relativfrequenz", "object_humidity", "object_pressure", "object_temperature" from mqtt_consumer'
    )
    return data['mqtt_consumer']


def get_watering_sessions(relative_freq):
    '''detects times in which the plant has been watered by measuring change(object_relativfrequenz)
    in WATERING_TIME_S

    args:
        column of object_relativfrequenz: pd.DataFrame

    assumptions:
        -Watering takes about 30seconds
            ->equals time difference of two consecutive data points
        -average fluctation of object_relativfrequenz values in WATERING_TIME_S is max=3%
            =MAX_NATURAL_FLUCTUATION_PERCENTAGE
            ->conclusion: everything above must be watering
        -watering=object_relativfrequenz increases, a drastic DECREASE indicates most likely
        a transmission error

    returns:
        list of timepoints, in which watering started, in the influxdb standard format

    problems:
        changing Min/Max also leads to drastic changes in object_relativfrequenz
            ->store history of min/max changes on sensor?
        outside plants: rain could last for longer than 30s
            ->results in several detected watering sessions

    use:
        - for further analysis of the data: changes due to watering are independent from all the
        other parameters
        - to optimize timing of watering sessions ()

    '''
    MAX_NATURAL_FLUCTUATION_PERCENTAGE = 3
    drastic_changes = relative_freq[relative_freq.diff() > MAX_NATURAL_FLUCTUATION_PERCENTAGE]
    return drastic_changes.index


def split_into_chunks(all_data, cut_points):
    tmin = pd.Timestamp.min.tz_localize('UTC')

    chunks = []
    prev_cut_point = tmin
    for point in cut_points:
        new_chunk = all_data[(prev_cut_point < all_data.index) & (all_data.index < point)]
        chunks.append(new_chunk)
        prev_cut_point = point
    chunks.append(all_data[prev_cut_point < all_data.index])

    return chunks


def get_cut_points(all_data):
    '''returns all cut points for split into chunks'''
    watering_sessions = get_watering_sessions(all_data)
    # TODO callibration_changes = get_callibration_changes(all_data)
    return watering_sessions


def get_callibration_changes(all_data):
    pass


def determine_regression_type(chunks):
    errors = np.zeros((len(chunks), len(FUNCS)))

    for id_chunk, chunk in enumerate(chunks):

        if len(chunk) < 5:
            # not enough datapoints for regression
            continue

        first = chunk.index[0]
        x = (chunk.index-first).total_seconds()
        y = chunk['object_relativfrequenz'].values

        params_list = []
        for id_func, f in enumerate(FUNCS):
            try:
                params = curve_fit(f, x, y)[0]
                params_list.append(params)

                errors[id_chunk][id_func] = (mean_squared_error(f(x, *params), y))
            except RuntimeError:
                # no fit found
                # TODO: what if several np.inf???->problem in min() as sum always = np.inf
                errors[id_chunk][id_func] = np.inf

    func_errors = errors.sum(axis=0)
    min_id = np.where(func_errors == min(func_errors))[0][0]

    """ chunk = chunks[-1]
    first = chunk.index[0]
    x = (chunk.index-first).total_seconds()
    plt.plot(x, chunk['object_relativfrequenz'].values)
    y = FUNCS[min_id](x, *param_test)
    plt.plot(x, y)
    plt.show() """
    chunk = chunks[-1]
    first = chunk.index[0]

    x = (chunk.index-first).total_seconds()
    plt.plot(x, chunk['object_relativfrequenz'].values)

    for id, func in enumerate(FUNCS[0:4]):
        y = func(x, *params_list[id])
        plt.plot(x, y)

    plt.show()

    return min_id


def visualize(chunk):
    plt.plot(chunk.index, chunk['object_relativfrequenz'])
    plt.show()


if __name__ == '__main__':
    # goal: find function which describes relativfrequenz(t) until next watering
    #           ->detect watering times
    #       how do the other parameters affect relativfrequenz(t)?
    #           ->ML model: find perfect choice for temperature and pressure so that
    #                       relativfrequenz(t) is as flat as possible
    #                       (its derivative's vals have to be as low as possible)

    all_data = get_data().dropna()
    relative_freq = all_data['object_relativfrequenz'].dropna()
    changes = get_watering_sessions(relative_freq)
    chunks = split_into_chunks(all_data, changes)
    # visualize(chunks[-1])
    reg_func_id = determine_regression_type(chunks)
    print(chunks)
