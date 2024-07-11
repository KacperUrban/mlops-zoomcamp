import pickle
import numpy as np
from flask import Flask, jsonify, request
import mlflow
from mlflow.tracking import MlflowClient
import os
import xgboost

RUN_ID = "1733bcf5fc484bdfb50461b0182158fd"
os.environ["AWS_PROFILE"] = "user1"

# Downloading dictvectorizer from AWS
TRACKING_SERVER_HOST = "ec2-3-75-179-164.eu-central-1.compute.amazonaws.com"
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
client = MlflowClient(f"http://{TRACKING_SERVER_HOST}:5000")

mlflow.set_experiment("nyc-taxi-experiment")

path = client.download_artifacts(run_id=RUN_ID, path='preprocessor', dst_path='.')

print(f"Downloading dictvectorizer to {path}")

with open('preprocessor/preprocessor.b', "rb") as f_out:
    dv = pickle.load(f_out)

# Downloading model from AWS
logged_model = f"runs:/{RUN_ID}/models"

model = mlflow.xgboost.load_model(logged_model)
print(f"Downloading model")

def prepare_features(ride):
    features = {}
    features['PU_Do'] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    X = dv.transform(features)
    X = xgboost.DMatrix(X)
    print(f"Transformed features: {X}")
    preds = model.predict(X)
    return preds[0]

app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def prediction_endpoint():
    try:
        ride = request.get_json()
        print(f"Received ride data: {ride}")

        features = prepare_features(ride)
        print(f"Prepared features: {features}")
        
        pred = predict(features)
        print(f"Prediction: {pred}")

        result = {
            'duration': np.round(float(pred),3),
            'model_version': RUN_ID
        }

        return jsonify(result)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)