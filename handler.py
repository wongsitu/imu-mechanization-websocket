try:
    import unzip_requirements
except ImportError:
    pass


import json
import numpy as np
import boto3
from botocore.config import Config

# from nav import Nav

# GRAVITY = 9.80665  # m / s ** 2

client = boto3.client(
    'apigatewaymanagementapi',
    endpoint_url="https://98ldqkpb7k.execute-api.us-east-1.amazonaws.com/dev",
    config=Config(tcp_keepalive=True),
)


def websocket_handler(event, context):
    route = event.get('requestContext', {}).get('routeKey')
    if route == '$connect':
        return {'statusCode': 200}
    elif route == '$disconnect':
        return {'statusCode': 200}
    elif route == '$default':
        message = event.get('body', {})

        connectionId = event.get('requestContext', {}).get('connectionId')
        message = event.get('body', {})

        # if 'nav' not in globals():
        #     set_nav(displacement, is_supercharged, drag_coeff)

        payload = {
            'fuel': np.random.rand(),
            'speed': np.random.rand(),
            'co2': np.random.rand(),
            'co': np.random.rand(),
            'nox': np.random.rand(),
            'unburned_hydrocarbons': np.random.rand(),
            'particulate': np.random.rand(),
        }

        client.post_to_connection(ConnectionId=connectionId, Data=json.dumps(payload).encode('utf-8'))
        return {'statusCode': 200}
    else:
        return {'statusCode': 400, 'body': 'Unknown WebSocket event'}


# def set_nav(displacement=None, is_supercharged=None, drag_coeff=None):
#     global nav
#     nav = Nav(
#         smoothing_critical_freq=0.03,
#         vz_depth=3,
#         period=0.01,
#         algo='madgwick',
#         smooth_fc=True,
#         fc_smoothing_critical_freq=0.02,
#         imu_damping=0.05,
#         fc_reduction_factor=0.5,
#         displacement=displacement,
#         is_supercharged=is_supercharged,
#         drag_coeff=drag_coeff,
#     )


# def set_params(displacement, is_supercharged, drag_coeff=None):
#     if 'nav' not in globals():
#         print('WARNING: Navigation not initialized. Call set_nav to initialize.')
#         return
#     nav.set_vehicle_params(displacement, is_supercharged, drag_coeff)


# def run_nav(batch):
#     if 'nav' not in globals():
#         print('WARNING: Navigation not initialized. Call set_nav to initialize.')
#         return

#     for data in batch:
#         t, ax, ay, az, ax_nog, ay_nog, az_nog, gx, gy, gz, mx, my, mz, lat, long, alt, heading, speed = data

#         acc = np.array([ax * GRAVITY, ay * GRAVITY, az * GRAVITY])
#         acc_nog = np.array([ax_nog * GRAVITY, ay_nog * GRAVITY, az_nog * GRAVITY])
#         gyr = np.array([gx, gy, gz])
#         mag = np.array([mx, my, mz])

#         nav.process_imu_update(t, acc, acc_nog, gyr, mag)
#         if lat is not None:
#             nav.process_gps_update(t, lat, long, alt, heading, speed)

#     fuel = nav.get_fuel(return_totals=False)
#     emissions = nav.get_emissions(return_totals=False)
#     motion = nav.get_motion()

#     return fuel | emissions


# def end_nav():
#     global nav
#     trip_metrics = nav.get_trip_metrics()
#     del nav
#     return trip_metrics
