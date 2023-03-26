import json
# import numpy as np
# from nav import Nav

# GRAVITY = 9.80665  # m / s ** 2


def websocket_handler(event, context):
    # connection_id = event['requestContext']['connectionId']
    # domain_name = event['requestContext']['domainName']
    # stage = event['requestContext']['stage']

    # if event['requestContext']['eventType'] == 'CONNECT':
    #     # Handle connect event
    #     pass
    # elif event['requestContext']['eventType'] == 'DISCONNECT':
    #     # Handle disconnect event
    #     pass
    print('It works')

    return {
        'statusCode': 200,
        'body': json.dumps({'message': 'success'})
    }


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
