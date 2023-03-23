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
min_len = min(len(acc), len(acc_nog), len(gyr), len(mag), start + 1000000000)
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
    initial_period=0.01,
    algo='madgwick',
    displacement=2.2,
    is_supercharged=False,
    drag_coeff=0.0038,
)


def run_nav(nav, data_batch):
    for data in data_batch:
        timestamp, ax, ay, az, ax_nog, ay_nog, az_nog, gx, gy, gz, mx, my, mz, lat, long, alt, heading, speed = data

        acc = np.array([[ax], [ay], [az]]) * GRAVITY
        acc_nog = np.array([[ax_nog], [ay_nog], [az_nog]]) * GRAVITY
        gyr = np.array([[gx], [gy], [gz]])
        mag = np.array([[mx], [my], [mz]])

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
sns.lineplot(fc[:, 0], label='fc')
sns.lineplot(fc[:, 1], label='emissions')

plt.figure()
sns.lineplot(fc[:, 2], label='total fc')
sns.lineplot(fc[:, 3], label='total emissions')

plt.figure()
v = np.array(v)
sns.lineplot(v[:, 0], label='vx')
sns.lineplot(v[:, 1], label='vy')
# sns.lineplot(v[:, 2], label='vz')
# plt.ylim(23, 25)

plt.figure()
a = np.array(a)
sns.lineplot(a[:, 0], label='ax')
sns.lineplot(a[:, 1], label='ay')
sns.lineplot(a[:, 2], label='az')

plt.show()
