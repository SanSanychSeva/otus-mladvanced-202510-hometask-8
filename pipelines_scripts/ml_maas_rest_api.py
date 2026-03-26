from flask import Flask, jsonify, request

ml_maas_rest_api = Flask(__name__)

@ml_maas_rest_api.route('/')
def health_check():
    return jsonify({'status':'MLaaS is online OK!'})

if __name__ == "__main__":
    ml_maas.run()