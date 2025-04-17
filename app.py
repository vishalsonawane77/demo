from flask import Flask, render_template, request
import pickle
import numpy as np
import os
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Get absolute paths to model files
def get_model_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)

# Load model and encoder
with open(get_model_path('crop_model.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(get_model_path('label_encoder.pkl'), 'rb') as f:
    le = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        N = int(request.form['N'])
        P = int(request.form['P'])
        K = int(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_data)[0]
        crop_name = le.inverse_transform([prediction])[0]

        image_path = f"static/crop_images/{crop_name.lower()}.jpg"
        full_image_path = os.path.join(app.static_folder, image_path)
        crop_image = image_path if os.path.exists(full_image_path) else None

        return render_template('result.html', 
                             crop=crop_name, 
                             image=crop_image,
                             N=N, P=P, K=K,
                             temperature=temperature,
                             humidity=humidity,
                             ph=ph,
                             rainfall=rainfall)
    except Exception as e:
        return f"Error: {str(e)}", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)), debug=False)
