from flask import Flask, request, jsonify
from tensorflow import keras
from PIL import Image
import cv2
import numpy as np

app = Flask(__name__)
model = keras.models.load_model("effnet.h5")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        img = Image.open(file)
        prediction = make_prediction(img)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

def make_prediction(img):
    # Preprocess and use the loaded model to make predictions
    # Your code for image preprocessing and prediction here
    # Example code:
    opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencvImage, (150, 150))
    img = img.reshape(1, 150, 150, 3)
    p = model.predict(img)
    p = np.argmax(p, axis=1)[0]
    return p

if __name__ == '__main__':
    app.run(debug=True)
