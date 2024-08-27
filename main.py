import requests
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


def preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_scale = np.expand_dims(img_array, axis=0) / 255.0
    return img_scale

app = Flask(__name__)

model = load_model('model/cats_dogs_cls.h5')
# image = preprocess_image('datasets/val/dogs/dog.5.jpg', target_size=(128, 128))
# prediction = model.predict(image)
# print(prediction[0][0])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        # Preprocess the image
        image = preprocess_image(file, target_size=(128, 128))

        # Make prediction
        prediction = model.predict(image)
        predicted_class = "Dog" if prediction[0][0] > 0.5 else "Cat"

        return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)