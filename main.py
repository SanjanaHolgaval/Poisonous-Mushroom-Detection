import os
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  # type: ignore
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
from PIL import Image

# Define directories
dataset_dir = 'dataset'
split_dir = 'split_data'
model_path = 'model/mushroom_model.h5'
results_dir = 'results'

# Function to check if image is valid
def is_image_valid(filepath):
    try:
        img = Image.open(filepath)
        img.load()
        return True
    except (IOError, SyntaxError):
        return False

# Function to split data
def split_data(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    for class_name in ['edible', 'poisonous']:
        class_dir = os.path.join(source_dir, class_name)
        all_images = os.listdir(class_dir)
        images = [img for img in all_images if is_image_valid(os.path.join(class_dir, img))]
        print(f"Valid images in {class_name}: {len(images)} (out of {len(all_images)})")
        # Split
        train_images, temp_images = train_test_split(images, test_size=val_ratio + test_ratio, random_state=42)
        val_images, test_images = train_test_split(temp_images, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)
        # Move to train
        for img in train_images:
            shutil.move(os.path.join(class_dir, img), os.path.join(dest_dir, 'train', class_name, img))
        for img in val_images:
            shutil.move(os.path.join(class_dir, img), os.path.join(dest_dir, 'val', class_name, img))
        for img in test_images:
            shutil.move(os.path.join(class_dir, img), os.path.join(dest_dir, 'test', class_name, img))

# Call split
# Ensure directories exist before splitting
os.makedirs(os.path.join(split_dir, 'train', 'edible'), exist_ok=True)
os.makedirs(os.path.join(split_dir, 'train', 'poisonous'), exist_ok=True)
os.makedirs(os.path.join(split_dir, 'val', 'edible'), exist_ok=True)
os.makedirs(os.path.join(split_dir, 'val', 'poisonous'), exist_ok=True)
os.makedirs(os.path.join(split_dir, 'test', 'edible'), exist_ok=True)
os.makedirs(os.path.join(split_dir, 'test', 'poisonous'), exist_ok=True)
# split_data(dataset_dir, split_dir)  # Commented out since data is already split

# Preprocessing
img_size = (224, 224)
batch_size = 32

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(split_dir, 'train'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(split_dir, 'val'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(split_dir, 'test'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the Model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator
)

# Save best model
model.save(model_path)

# Print class indices to verify
print("\n=== CLASS INDICES ===")
print(f"Train Generator Classes: {train_generator.class_indices}")
print(f"Validation Generator Classes: {val_generator.class_indices}")
print(f"Test Generator Classes: {test_generator.class_indices}")
print("Note: 0=edible, 1=poisonous")

# Evaluate Model
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc}')

# Predictions
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int).flatten()
true_classes = test_generator.classes

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
print('Confusion Matrix:')
print(cm)

# Classification Report
print('Classification Report:')
print(classification_report(true_classes, predicted_classes, target_names=['edible', 'poisonous']))

# Plot Results
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,3,2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.title('Loss')

plt.subplot(1,3,3)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0,1], ['edible', 'poisonous'])
plt.yticks([0,1], ['edible', 'poisonous'])

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'results.png'))
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(true_classes, predictions)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig(os.path.join(results_dir, 'roc.png'))
plt.show()

# Prediction System
def predict_mushroom(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    confidence = prediction[0][0]
    if confidence > 0.5:
        return 'Edible', confidence
    else:
        return 'Poisonous', 1 - confidence

# Example usage (uncomment to test)
# result, conf = predict_mushroom('path/to/mushroom_image.jpg')
# print(f'Prediction: {result}, Confidence: {conf}')