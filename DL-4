import os
import zipfile
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Function to download and extract the dataset
def download_dataset(url, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    response = requests.get(url, stream=True)
    with open(os.path.join(target_dir, 'pistachio_dataset.zip'), 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
    with zipfile.ZipFile(os.path.join(target_dir, 'pistachio_dataset.zip'), 'r') as zip_ref:
        zip_ref.extractall(target_dir)

# Download and extract the dataset
dataset_url = 'https://www.muratkoklu.com/datasets/pistachio_dataset.zip'
download_dataset(dataset_url, 'pistachio_dataset')

# Define directories for train and validation data
train_dir = 'pistachio_dataset/train'
validation_dir = 'pistachio_dataset/validation'

# Define constants
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 2  # Assuming 2 classes for pistachio dataset (pistachio and non-pistachio)

# Load and preprocess the data
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# Define the CNN architectures (AlexNet, VGG16, VGG19)
def alexnet():
    model = Sequential([
        Conv2D(96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        MaxPooling2D(pool_size=(3,3), strides=(2,2)),
        Conv2D(256, kernel_size=(5,5), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3,3), strides=(2,2)),
        Conv2D(384, kernel_size=(3,3), padding='same', activation='relu'),
        Conv2D(384, kernel_size=(3,3), padding='same', activation='relu'),
        Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3,3), strides=(2,2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

def vgg16():
    model = Sequential([
        Conv2D(64, kernel_size=(3,3), padding='same', activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'),
        Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'),
        Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'),
        Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'),
        Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'),
        Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'),
        Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'),
        Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

def vgg19():
    model = Sequential([
        Conv2D(64, kernel_size=(3,3), padding='same', activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'),
        Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'),
        Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'),
        Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'),
        Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'),
        Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'),
        Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'),
        Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'),
        Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'),
        Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'),
        Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# Choose the model to train and compile it
model = vgg16()  # Change to alexnet() or vgg19() for different architectures
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, steps_per_epoch=train_generator.samples // BATCH_SIZE, 
                    epochs=EPOCHS, validation_data=validation_generator, 
                    validation_steps=validation_generator.samples // BATCH_SIZE)

# Save the trained model
model.save('pistachio_cnn_model.h5')

# Evaluate the model
test_loss, test_acc = model.evaluate(validation_generator, verbose=2)
print(f'Test accuracy: {test_acc}')

# Visualize training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
