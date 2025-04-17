from flask import Flask, render_template, request
import pickle
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# Load model and encoder
with open('crop_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
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
            crop_image = image_path if os.path.exists(image_path) else None

            return render_template('result.html', crop=crop_name, image=crop_image,
                                   N=N, P=P, K=K, temperature=temperature,
                                   humidity=humidity, ph=ph, rainfall=rainfall)
        except Exception as e:
            return f"Error: {e}"
    return render_template('index.html')
    
if __name__ == '__main__':
    app.run(debug=True)
