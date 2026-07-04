import os
import tensorflow as tf
from tensorflow.keras import layers, models

print("Loading MNIST dataset...")
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0

# Reshape data to fit the CNN model (28x28 pixels, 1 color channel)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Build Convolutional Neural Network (CNN) model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
print("Training the model...")
model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Save the trained model
model.save('models/digit_model.h5')
print("Model saved successfully as 'models/digit_model.h5'!")