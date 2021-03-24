import numpy as np
import pandas as pd

features = ['current_speed', 'use', 'depth', 'salinity', 'shore_distance']

def failure_rate(t, current_speed, use, depth, salinity, shore_distance=None):
    C = 0.00002
    a = 2
    b = 0.00005
    c = 0.03
    d = 1
    return C * (a*current_speed*t + b*use*t**3 + c*depth*t**2 + d*salinity*t)

def create_population(size):
    population = pd.DataFrame(
        np.random.rand(size, len(features)),
        columns=features
    )
    return population

def create_sensor_record(sample):
    x = np.linspace(0, 1000, 1000)
    y = failure_rate(x, **sample)
    return pd.DataFrame({
        't':x,
        'sensor_reading':y
    })