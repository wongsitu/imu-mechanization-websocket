try:
    import unzip_requirements
except ImportError:
    pass

import json
import numpy as np
import boto3
from botocore.config import Config

from nav import Nav

GRAVITY = 9.80665  # m / s ** 2
DEG_TO_RAD = np.pi / 180

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
        message = json.loads(message)

        if 'nav' not in globals():
            set_nav(message['displacement'], message['isSupercharged'], message['drag'])

        acc = np.array(
            [
                message['accelerometerWithGravity']['x'],
                message['accelerometerWithGravity']['y'],
                message['accelerometerWithGravity']['z'],
            ]
        )

        acc_nog = np.array(
            [
                message['accelerometerWithoutGravity']['x'],
                message['accelerometerWithoutGravity']['y'],
                message['accelerometerWithoutGravity']['z'],
            ]
        )

        gyro = np.array(
            [
                message['gyroscope']['beta'] * DEG_TO_RAD,
                message['gyroscope']['gamma'] * DEG_TO_RAD,
                message['gyroscope']['alpha'] * DEG_TO_RAD,
            ]
        )

        mag = np.array(
            [
                message['magnetometer']['x'],
                message['magnetometer']['y'],
                message['magnetometer']['z'],
            ]
        )

        payload = run_nav(
            message['time'],
            acc,
            acc_nog,
            gyro,
            mag,
            message['lat'],
            message['lon'],
            message['alt'],
            message['heading'],
            message['speed'],
        )

        # payload = {
        #     'fuel': np.random.rand(),
        #     'speed': np.random.rand(),
        #     'co2': np.random.rand(),
        #     'co': np.random.rand(),
        #     'nox': np.random.rand(),
        #     'unburned_hydrocarbons': np.random.rand(),
        #     'particulate': np.random.rand(),
        # }

        client.post_to_connection(ConnectionId=connectionId, Data=json.dumps(payload).encode('utf-8'))
        return {'statusCode': 200}
    else:
        return {'statusCode': 400, 'body': 'Unknown WebSocket event'}


def set_nav(displacement=None, is_supercharged=None, drag_coeff=None):
    global nav
    nav = Nav(
        smoothing_critical_freq=0.03,
        vz_depth=3,
        period=0.01,  ############################ UPDATE THIS #######################################
        algo='madgwick',
        smooth_fc=True,
        fc_smoothing_critical_freq=0.02,
        imu_damping=0.05,
        fc_reduction_factor=0.5,
        displacement=displacement,
        is_supercharged=is_supercharged,
        drag_coeff=drag_coeff,
    )


def set_params(displacement, is_supercharged, drag_coeff=None):
    if 'nav' not in globals():
        print('WARNING: Navigation not initialized. Call set_nav to initialize.')
        return
    nav.set_vehicle_params(displacement, is_supercharged, drag_coeff)


def run_nav(t, acc, acc_nog, gyro, mag, lat, long, alt, heading, speed):
    if 'nav' not in globals():
        print('WARNING: Navigation not initialized. Call set_nav to initialize.')
        return

    nav.process_imu_update(t, acc, acc_nog, gyro, mag)
    if lat is not None:
        nav.process_gps_update(t, lat, long, alt, heading, speed)

    fuel = nav.get_fuel(return_totals=False)
    emissions = nav.get_emissions(return_totals=False)
    speed = nav.get_motion(speed_only=True)

    return {**fuel, **emissions, **speed}


def end_nav():
    global nav
    trip_metrics = nav.get_trip_metrics()
    del nav
    return trip_metrics
