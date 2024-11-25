from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import os

from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow CORS for all routes


app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('landmark_recognition_resnet50_finetuned.h5')

# Load label names
label_names = ["all_souls", "ashmolean", "ballol", "bodleian", "christ_church", "cornmarket",
               "hertford", "jesus", "magdalen", "new", "oriel", "oxford", "pitt_rivers",
               "radcliffe", "trinity", "worcester", "keble"]

def prepare_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image_path = os.path.join('uploads', file.filename)
    file.save(image_path)

    # Prepare the image
    image = prepare_image(image_path)

    # Predict using the model
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    class_name = label_names[class_index]

    # Construct similar images paths
    similar_images = []
    for i in range(1, 4):
        image_filename = f'{class_name.lower().replace(" ", "_")}_{i}.jpg'
        image_path = os.path.join('static', 'images', image_filename)
        if os.path.exists(image_path):
            similar_images.append(f'/static/images/{image_filename}')
        else:
            print(f'Warning: {image_filename} does not exist.')

    return jsonify({
        'landmark_name': class_name,
        'similar_images': similar_images
    })



if __name__ == '__main__':
    # Create uploads folder if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)
