from nav import Nav
import numpy as np


import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

datafile = 'data/from_bragg'

acc = pd.read_csv(os.path.join(datafile, 'AccelerometerUncalibrated.csv'))
acc_nog = pd.read_csv(os.path.join(datafile, 'Accelerometer.csv'))
gyr = pd.read_csv(os.path.join(datafile, 'Gyroscope.csv'))
mag = pd.read_csv(os.path.join(datafile, 'Magnetometer.csv'))
gps = pd.read_csv(os.path.join(datafile, 'Location.csv'))

min_len = min(len(acc), len(acc_nog), len(gyr), len(mag), 10000)
timestamps = acc.seconds_elapsed.to_numpy()[:min_len]
acc = acc[['x', 'y', 'z']].to_numpy()[:min_len]
acc_nog = acc_nog[['x', 'y', 'z']].to_numpy()[:min_len]
gyr = gyr[['x', 'y', 'z']].to_numpy()[:min_len]
mag = mag[['x', 'y', 'z']].to_numpy()[:min_len]
gps = gps[['seconds_elapsed', 'latitude', 'longitude', 'altitude', 'bearing', 'speed']].to_numpy()


nav = Nav(
    smoothing_critical_freq=0.03,
    vz_depth=3,
    initial_period=0.01,
    algo='ekf',
    displacement=3.4,
    is_supercharged=False,
    drag_coeff=0.0038,
)



def run_nav(nav, data_batch):
    for data in data_batch:
        timestamp, ax, ay, az, ax_nog, ay_nog, az_nog, gx, gy, gz, mx, my, mz, lat, long, alt, heading, speed = data

        acc = np.array([[ax], [ay], [az]])
        acc_nog = np.array([[ax_nog], [ay_nog], [az_nog]])
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
# for i, t in enumerate(timestamps):
#     if i % 1000 == 0:
#         print(i / len(timestamps))
#     data = [t, *acc[i], *acc_nog[i], *gyr[i], *mag[i], None, None, None, None, None]
#     if gps_index < len(gps) and t >= gps[gps_index][0]:
#         data[-5:] = [*gps[gps_index][1:]]
#         gps_index += 1
#         batch.append(data)
#         continue
#     batch.append(data)

#     if len(batch) >= 20:
#         run_nav(nav, batch)
#         batch = []
#         fc.append(nav.get_fuel_consumption())
#         vel_acc = nav.get_motion()
#         v.append(vel_acc[0].flatten())
#         a.append(vel_acc[1].flatten())


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
    fc.append(nav.get_fuel_consumption())
    vel_acc = nav.get_motion()
    v.append(vel_acc[0].flatten())
    a.append(vel_acc[1].flatten())


plt.figure()
fc = np.array(fc)
sns.lineplot(fc[:, 0], label='fc')
sns.lineplot(fc[:, 1], label='emissions')

plt.figure()
v = np.array(v)
sns.lineplot(v[:, 0], label='vx')
sns.lineplot(v[:, 1], label='vy')
sns.lineplot(v[:, 2], label='vz')

plt.figure()
a = np.array(a)
sns.lineplot(a[:, 0], label='ax')
sns.lineplot(a[:, 1], label='ay')
sns.lineplot(a[:, 2], label='az')

plt.show()
