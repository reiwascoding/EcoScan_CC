from flask import Flask, request, jsonify
import tensorflow as tf
from urllib.request import urlopen
import numpy as np
from google.cloud import storage
import requests
from io import BytesIO

app = Flask(__name__)

# Load the H5 model
def load_model():
    client = storage.Client()
    bucket = client.get_bucket('ecoscan')
    blob = storage.Blob('models/model-waste-classification.h5', bucket)
    blob.download_to_filename('/tmp/model-waste-classification.h5')

    model = tf.keras.models.load_model('/tmp/model-waste-classification.h5')
    return model
# Preprocess the image
def preprocess_image(image):
    # Resize the image
    resized_image = tf.image.resize(image, (224, 224))

    # Convert the image to an array
    image_array = np.array(resized_image)

    # Normalize the image
    normalized_image = image_array / 255.0

    # Add an extra dimension to match the model's input shape
    preprocessed_image = np.expand_dims(normalized_image, axis=0)

    return preprocessed_image

# Classify the image using the loaded model
def classify_image(image):
    # Load the model
    model = load_model()

    # Make predictions
    predictions = model.predict(image)

    # Process the predictions
    predicted_class = np.argmax(predictions, axis=1)
    class_label = 'organic' if predicted_class == 0 else 'non-organic'

    return class_label

# API endpoint for image classification
@app.route('/classify', methods=['POST'])
def classify():
    if 'image' in request.files:
        # Image from local file
        image = request.files['image']
        image_data = image.read()
    elif 'image_url' in request.form:
        # Image from URL
        image_url = request.form['image_url']
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to retrieve the image from the provided URL.'}), 400
        image_data = response.content
    else:
        return jsonify({'error': 'No image or image URL provided.'}), 400

    # Convert the image data to TensorFlow Tensor
    image = tf.image.decode_jpeg(image_data, channels=3)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Classify the image
    class_label = classify_image(preprocessed_image)

    return jsonify({'class': class_label})

@app.route('/')
def health_check():
    return 'OK'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    app.run(debug=True)