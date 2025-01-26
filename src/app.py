from flask import Flask, request, jsonify
import pickle
import numpy as np
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model
MODEL_PATH = 'src/models/iris_model.pkl'
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

@app.before_request
def log_request_info():
    logger.info(f"Request Headers: {request.headers}")
    logger.info(f"Request Body: {request.get_data()}")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict the iris species.
    Input: JSON containing 'features' (a list of 4 numerical values).
    Output: Predicted class and confidence score.
    """
    try:
        # Parse input JSON
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)

        return jsonify({
            'predicted_class': int(prediction[0]),
            'confidence': prediction_proba.tolist()
        })
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)