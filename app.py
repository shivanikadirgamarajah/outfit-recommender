from flask import Flask, request, jsonify, render_template
from joblib import load 
import numpy as np 

app = Flask(__name__)
model = load("outfit_model.joblib")
label_encoder = load("label_encoder.joblib")

@app.route('/')
def serve_html():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [data['temperature'],data['humidity'], data['wind_speed'], data['precipitation']]
    prediction = model.predict([features])[0]
    outfit = label_encoder.inverse_transform([prediction])[0]
    return jsonify({'recommended_outfit': outfit})

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)