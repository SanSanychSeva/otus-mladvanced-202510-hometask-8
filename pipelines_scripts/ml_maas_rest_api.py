'''
Script: ml_maas_rest_api.py
Description: 
    This script implements a REST API for a Machine Learning as a Service (MLaaS)
'''

import pandas as pd
from flask import Flask, jsonify, request

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
    except:
        return jsonify({'error': 'Invalid JSON data in request body'}), 400

    try:
        new_data = pd.DataFrame(data)
    except:
        return jsonify({'error': 'Error occurred while processing the data'}), 400

    try:
        pred = load_model_n_predict(new_data)
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