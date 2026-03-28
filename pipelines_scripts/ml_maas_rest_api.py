'''
Script: ml_maas_rest_api.py
Description: 
    This script implements a REST API for a Machine Learning as a Service (MLaaS)
'''

import pandas as pd
from flask import Flask, jsonify, request
from loguru import logger

from src.stand_alone_inference import load_model_n_predict

ml_maas_rest_api = Flask(__name__)

@ml_maas_rest_api.route('/')
def health_check():
    return jsonify({'service status':'MLaaS is online OK!'})

@ml_maas_rest_api.route("/submit_analysis", methods=["POST"])
def prediction():
    '''
    uses the model to predict the diagnosis based on the data sent in the request body, 
    and returns the diagnosis as a response (both in and out data have json format)
    '''
    try:
        data = request.get_json()
        logger.debug(
            f"Flask service 'submit_analysis' calls function 'prediction': reporting received data: {data}"
            )
        for k in data.keys():
            data[k] = [data[k]] 
    except:
        return jsonify({'error': 'Invalid JSON data in request body'}), 400

    try:
        new_data = pd.DataFrame(data)
        logger.debug(
            "Flask service 'submit_analysis' / function 'prediction': reporting data converted to DataFrame"
            )
    except:
        return jsonify({'error': 'Error occurred while converting JSON data to DataFrame'}), 400

    try:
        pred = load_model_n_predict(new_data)
        logger.debug(
            f"Flask service 'submit_analysis' / function 'prediction': reporting prediction result: {pred}"
            )
    except:
        return jsonify({'error': 'either bad values or invalid field names - consult service description'}
                       ), 400

    if pred == 0:
        diag = 'No cancer detected - continue regular monitoring'
    else:
        diag = 'Cancer is likely - report to your doctor immediately'

    return jsonify({'diagnosis': diag})

if __name__ == "__main__":
    ml_maas_rest_api.run()