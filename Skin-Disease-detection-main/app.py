from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  
# Load Model
MODEL_PATH = "model_resnet50.h5"
model = load_model(MODEL_PATH)

# Define class labels
CLASS_LABELS = ['Acne', 'Actinic keratosis', 'Atopic Dermatitis', 'Basal Cell Carcinoma', 'Benign keratosis',
                'Chickenpox', 'Dermatofibroma', 'Eczemaa', 'Hairloss', 'Impetigo', 'Infectious erythema',
                'Melanocytic nevus', 'Melanoma', 'Nail Fungus', 'Normal', 'Rosacea', 'Scabies', 'Skin Allergy',
                'Skin warts', 'Squamous cell carcinoma', 'Tinea Ringworm Candidiasis']

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Read image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (100, 100))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    
    # Prediction
    preds = model.predict(img)[0] * 100
    max_index = np.argmax(preds)
    predicted_label = CLASS_LABELS[max_index]
    confidence = preds[max_index]
    
    return jsonify({"prediction": predicted_label, "confidence": round(float(confidence), 2)})

if __name__ == '__main__':
    app.run(debug=True)
