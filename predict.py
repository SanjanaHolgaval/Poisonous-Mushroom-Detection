import sys
import os
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore

MODEL_PATH = os.path.join('model', 'mushroom_model_improved.keras')
IMG_SIZE = (224, 224)

if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}. Run convert_model.py first.")
    raise SystemExit(1)

model = load_model(MODEL_PATH)

if len(sys.argv) < 2:
    print("Usage: python predict.py <image_path>")
    raise SystemExit(1)

img_path = sys.argv[1]
if not os.path.exists(img_path):
    print(f"Image not found: {img_path}")
    raise SystemExit(1)

img = load_img(img_path, target_size=IMG_SIZE)
arr = img_to_array(img) / 255.0
arr = np.expand_dims(arr, 0)

pred = model.predict(arr)[0]
# If model output single sigmoid, pred may be scalar or array of one
if hasattr(pred, '__len__'):
    score = float(pred[0])
else:
    score = float(pred)

label = 'edible' if score > 0.5 else 'poisonous'
confidence = score if label == 'edible' else 1 - score
print(f"Image: {img_path}")
print(f"Predicted: {label} (confidence={confidence:.4f})")
