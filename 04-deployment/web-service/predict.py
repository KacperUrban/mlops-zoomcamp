import pickle
import numpy as np
from flask import Flask, jsonify, request

with open('lin_reg.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

def prepare_features(ride):
    features = {}
    features['PU_Do'] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    X = dv.transform(features)
    print(f"Transformed features: {X}")
    preds = np.round(model.predict(X), 3)
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
            'duration': pred
        }

        return jsonify(result)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)