import os
import io
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__, template_folder='templateFiles', static_folder='staticFiles')

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

data_directory = "/Users/krishuppal/Desktop/SIH_NN"
classes = os.listdir(data_directory)
class_to_label = {cls: idx for idx, cls in enumerate(classes)}

image_data_generator = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.3
)

batch_size = 32

train_generator = image_data_generator.flow_from_directory(
    data_directory,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = image_data_generator.flow_from_directory(
    data_directory,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

num_classes = len(classes)

class_labels = list(train_generator.class_indices.keys())

X_train = []
y_train = []
X_validation = []
y_validation = []

for i in range(len(train_generator)):
    batch_images, batch_labels = train_generator[i]
    X_train.extend(batch_images)
    y_train.extend(batch_labels)

for i in range(len(validation_generator)):
    batch_images, batch_labels = validation_generator[i]
    X_validation.extend(batch_images)
    y_validation.extend(batch_labels)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_validation = np.array(X_validation)
y_validation = np.array(y_validation)

y_train = np.argmax(y_train, axis=1)
y_validation = np.argmax(y_validation, axis=1)

from keras import layers, models
cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(2, 2), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters=64, kernel_size=(2, 2), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(5, activation='softmax')
])

cnn.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

cnn.fit(X_train, y_train, epochs=30)
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    # Check if the file is empty
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    # Initialize a response dictionary
    response_data = {}

    try:
        
        # Load and preprocess the uploaded image
        img_bytes = file.read()
        img = image.load_img(io.BytesIO(img_bytes), target_size=(128, 128))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # Normalize the image

        # Make a prediction
        prediction = cnn.predict(img)
        predicted_class = np.argmax(prediction)

        # Get the class label based on the predicted class
        class_labels = list(train_generator.class_indices.keys())
        predicted_label = class_labels[predicted_class]

        # Add the prediction result to the response dictionary
        response_data['prediction'] = predicted_label

    except Exception as e:
        # If there's an error, add an error message to the response dictionary
        response_data['error'] = str(e)

    # Return the response data as JSON
    return jsonify(response_data)
# Update the index route to render the HTML form
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

