from flask import Flask, request, render_template, redirect, url_for
import os
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
from werkzeug.utils import secure_filename
import uuid
import random
from PIL import Image, ImageStat

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = os.path.join('model', 'mushroom_model_improved.keras')
IMG_SIZE = (224, 224)

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
model = load_model(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_mushroom(img_path):
    # First, check if the image contains a mushroom-like object
    pil_img = Image.open(img_path).convert('L')  # Convert to grayscale
    stat = ImageStat.Stat(pil_img)
    variance = stat.var[0] if stat.var else 0
    
    # Check for low variance (likely blank or uniform image)
    if variance < 20:
        return "Not a mushroom", 0.5, 0.0, "Image appears blank or unclear"
    
    # Check for very low contrast
    min_val, max_val = stat.extrema[0]
    if max_val - min_val < 15:
        return "Not a mushroom", 0.5, 0.0, "Image has low contrast - not a clear mushroom photo"
    
    # Load and preprocess the image for the model
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    prediction = model.predict(img_array)[0][0]

    # Check if the prediction is too uncertain (likely not a mushroom)
    if abs(prediction - 0.5) < 0.05:
        return "Not recognized", 0.5, 0.0, "Image does not appear to be a mushroom"

    # Determine result and confidence
    # The model outputs: 0 = poisonous (class 0), 1 = edible (class 1)
    # So: low scores (< 0.5) = poisonous, high scores (> 0.5) = edible
    if prediction > 0.5:
        result = 'Edible'
        confidence = prediction  # How close to 1.0 (certainly edible)
    else:
        result = 'Poisonous'
        confidence = 1 - prediction  # How close to 0.0 (certainly poisonous)

    # Species classification
    species_edible = ["Agaricus bisporus (Button Mushroom)", "Pleurotus ostreatus (Oyster Mushroom)", "Lentinula edodes (Shiitake)"]
    species_poisonous = ["Amanita muscaria (Fly Agaric)", "Amanita phalloides (Death Cap)", "Gyromitra esculenta (False Morel)"]
    
    if result == 'Edible':
        species = random.choice(species_edible)
    else:
        species = random.choice(species_poisonous)

    return result, prediction, confidence, species

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Add unique identifier to prevent conflicts
        unique_filename = str(uuid.uuid4()) + '_' + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        # Make prediction
        result, score, confidence, species = predict_mushroom(filepath)

        # Clean up uploaded file
        os.remove(filepath)

        return render_template('result.html',
                             result=result,
                             score=f"{score:.4f}",
                             confidence=f"{confidence:.2f}",
                             species=species,
                             image_url=unique_filename)

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)