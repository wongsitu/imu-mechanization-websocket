# IMU Mechanization Serverless WebSocket

## Description
This repository contains IMU mechanization and IMU/GNSS/magentometer sensor fusion code to compute navigation parameters, fuel consumption, and CO2 emissions of a vehicle given data from a smartphone contained within it.  This code is part of the Fuellytics application, and is run through the AWS Lambdda serverless computing platform, connecting to the [Fuellytics backend](https://github.com/wongsitu/fuellytics_backend) via the WebSocket protocol.  The main frontend repository of the Fuellytics application is available [here](https://github.com/adamreidsmith/fuellytics).

URL:

```
wss://2m72fvzj25.execute-api.us-east-1.amazonaws.com/dev
```

## Setup

```
npm install
```

```
python -m venv .env
```

```
source .env/bin/activate
```

```
pip install -r requirements.txt
```
---
Made with :heart: by Chavisa Sornsakul, Wai Ka Wong Situ, and Adam Smith.
