from nav import Nav
import numpy as np

GRAVITY = 9.80665


import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

datafile = '../fuellytics_test/data/from_bragg'

acc = pd.read_csv(os.path.join(datafile, 'AccelerometerUncalibrated.csv'))
acc_nog = pd.read_csv(os.path.join(datafile, 'Accelerometer.csv'))
gyr = pd.read_csv(os.path.join(datafile, 'Gyroscope.csv'))
mag = pd.read_csv(os.path.join(datafile, 'Magnetometer.csv'))
gps = pd.read_csv(os.path.join(datafile, 'Location.csv'))

start = 0
min_len = min(len(acc), len(acc_nog), len(gyr), len(mag), start + 5000000)
timestamps = acc.seconds_elapsed.to_numpy()[start:min_len]
acc = acc[['x', 'y', 'z']].to_numpy()[start:min_len]
acc_nog = acc_nog[['x', 'y', 'z']].to_numpy()[start:min_len]
gyr = gyr[['x', 'y', 'z']].to_numpy()[start:min_len]
mag = mag[['x', 'y', 'z']].to_numpy()[start:min_len]
gps = gps[['seconds_elapsed', 'latitude', 'longitude', 'altitude', 'bearing', 'speed']]
gps = gps[gps.seconds_elapsed > timestamps[0]].to_numpy()


nav = Nav(
    smoothing_critical_freq=0.03,
    vz_depth=3,
    period=0.01,
    algo='madgwick',
    displacement=2.378,
    is_supercharged=False,
    drag_coeff=0.000641 / 0.07803855,
    smooth_fc=True,
    fc_smoothing_critical_freq=0.02,
    imu_damping=0.05,
    fc_reduction_factor=0.5,
)


def run_nav(nav, data_batch):
    for data in data_batch:
        timestamp, ax, ay, az, ax_nog, ay_nog, az_nog, gx, gy, gz, mx, my, mz, lat, long, alt, heading, speed = data

        acc = np.array([ax, ay, az]) * GRAVITY
        acc_nog = np.array([ax_nog, ay_nog, az_nog]) * GRAVITY
        gyr = np.array([gx, gy, gz])
        mag = np.array([mx, my, mz])

        nav.process_imu_update(timestamp, acc, acc_nog, gyr, mag)
        if lat is not None:
            nav.process_gps_update(timestamp, lat, long, alt, heading, speed)


################# SYSTEM TEST ########################################################################
batch = []
gps_index = 1
fc = []
v = []
a = []
time = []

for i, t in enumerate(timestamps):
    if i % 1000 == 0:
        print(i / len(timestamps))
        print(nav.get_trip_metrics())

    data = [t, *acc[i], *acc_nog[i], *gyr[i], *mag[i], None, None, None, None, None]

    have_gps = False
    if gps_index < len(gps) and t >= gps[gps_index, 0]:
        data[-5:] = [*gps[gps_index][1:]]
        have_gps = True
        gps_index += 1

    run_nav(nav, [data])

    # if have_gps:
    fc.append(nav.get_fuel_and_emissions())
    vel_acc = nav.get_motion()
    v.append(vel_acc[0].flatten())
    a.append(vel_acc[1].flatten())


plt.figure()
fc = np.array(fc)
sns.lineplot(y=fc[:, 0], x=timestamps / 60, label='fc')
sns.lineplot(y=fc[:, 1], x=timestamps / 60, label='emissions')

plt.figure()
sns.lineplot(y=fc[:, 2], x=timestamps / 60, label='total fc')
sns.lineplot(y=fc[:, 3], x=timestamps / 60, label='total emissions')

plt.figure()
v = np.array(v)
sns.lineplot(y=v[:, 0], x=timestamps / 60, label='vx')
sns.lineplot(y=v[:, 1], x=timestamps / 60, label='vy')
sns.lineplot(y=v[:, 2], x=timestamps / 60, label='vz')

plt.figure()
a = np.array(a)
sns.lineplot(y=a[:, 0], x=timestamps / 60, label='ax')
sns.lineplot(y=a[:, 1], x=timestamps / 60, label='ay')
sns.lineplot(y=a[:, 2], x=timestamps / 60, label='az')

plt.show()


import time

a = 15.3458
b = 0.3463256

t = time.perf_counter()
for _ in range(100000):
    c = a * b
    d = a * b
print(time.perf_counter() - t)

t = time.perf_counter()
for _ in range(100000):
    ab = a * b
    c = ab
    d = ab
print(time.perf_counter() - t)
