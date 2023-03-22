import json


def connect(event, context):
    connection_id = event['requestContext']['connectionId']
    print(f"New connection: {connection_id}")
    return {"statusCode": 200}


def default(event, context):
    connection_id = event['requestContext']['connectionId']
    body = json.loads(event['body'])
    message = body['message']
    print(f"Received message: {message} from {connection_id}")
    return {"statusCode": 200}
