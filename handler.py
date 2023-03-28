try:
    import unzip_requirements
except ImportError:
    pass


import json
import numpy as np
import boto3
# from nav import Nav

# GRAVITY = 9.80665  # m / s ** 2

client = boto3.client('apigatewaymanagementapi', endpoint_url="https://98ldqkpb7k.execute-api.us-east-1.amazonaws.com/dev")

def websocket_handler(event, context):
    route = event.get('requestContext', {}).get('routeKey')
    if route == '$connect':
        return {'statusCode': 200 }
    elif route == '$disconnect':
        return {'statusCode': 200 }
    elif route == '$default':
        message = event.get('body', {})
        print(message)

        connectionId = event.get('requestContext', {}).get('connectionId')
        payload= { 'fuelConsumption': 10, 'co2Emissions': 0, 'n2oEmissions': 0, 'ch4Emissions': 0 }
        client.post_to_connection(ConnectionId=connectionId, Data=json.dumps(payload).encode('utf-8'))
        return {'statusCode': 200 }
    else:
        return {'statusCode': 400, 'body': 'Unknown WebSocket event'}


# def connect(event, context):
#     connection_id = event['requestContext']['connectionId']
#     print(f"New connection: {connection_id}")
#     return {"statusCode": 200}


# def default(event, context):
#     connection_id = event['requestContext']['connectionId']
#     body = json.loads(event['body'])
#     message = body['message']
#     print(f"Received message: {message} from {connection_id}")

#     # Receive a message to start navigation -> call set_nav
#     # Set vehicle paramters with set_params (if not already set with set_nav)
#     # Receive a batch of IMU/Magnetometer/GPS data and pass it to run_nav. Return the fuel consumption and emissions data
#     # End the navigation with end_nav and return trip metrics

#     return {"statusCode": 200}


def set_nav(displacement=None, is_supercharged=None, drag_coeff=None):
    global nav
    nav = Nav(
        smoothing_critical_freq=0.03,
        vz_depth=3,
        period=0.01,
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


def run_nav(batch):
    if 'nav' not in globals():
        print('WARNING: Navigation not initialized. Call set_nav to initialize.')
        return

    for data in batch:
        t, ax, ay, az, ax_nog, ay_nog, az_nog, gx, gy, gz, mx, my, mz, lat, long, alt, heading, speed = data

        acc = np.array([ax * GRAVITY, ay * GRAVITY, az * GRAVITY])
        acc_nog = np.array(
            [ax_nog * GRAVITY, ay_nog * GRAVITY, az_nog * GRAVITY])
        gyr = np.array([gx, gy, gz])
        mag = np.array([mx, my, mz])

        nav.process_imu_update(t, acc, acc_nog, gyr, mag)
        if lat is not None:
            nav.process_gps_update(t, lat, long, alt, heading, speed)

    return nav.get_fuel_and_emissions()


def end_nav():
    global nav
    trip_metrics = nav.get_trip_metrics()
    del nav
    return trip_metrics
