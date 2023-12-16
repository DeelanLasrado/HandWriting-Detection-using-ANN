import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
print(X_train.shape)
# (60000, 28, 28) 60k-no of datasets    28*28- grid size
print(y_test)
#[7 2 1 ... 4 5 6]

# Display the first training image
plt.matshow(X_train[0], cmap='gray')
plt.show()

# Flatten the images
flat_X_train = X_train.reshape(-1, 28*28)
flat_X_test = X_test.reshape(-1, 28*28)
print(flat_X_train.shape)
# (60000, 784)

# Define a simple neural network model
model = keras.Sequential([
    keras.layers.Input(shape=(28*28,)),  # Adjust the input shape to match the flattened data
    keras.layers.Dense(100, activation='sigmoid'),
    keras.layers.Dense(50, activation='sigmoid'),
    keras.layers.Dense(20, activation='sigmoid')

])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(flat_X_train, y_train, epochs=10)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(flat_X_test, y_test)
print(f"Test accuracy: {test_accuracy}")

# Make predictions on the test data
y_pred = model.predict(flat_X_test)
print("The predicted value Z score\n  ",y_pred[-1])
print('the actual value  ',y_test[-1])
# Get the predicted label for the first test sample
first_test_sample_label = np.argmax(y_pred[-1])  #The argmax function is used to find the index of the maximum value along a specified axis of a NumPy array
print(f"Predicted label for the first test sample: {first_test_sample_label}")

# Display the first test image
plt.matshow(X_test[-1], cmap='gray')
plt.show()


# Import the necessary libraries
import numpy as np

# Get the predicted labels for all test samples
y_pred_labels = np.argmax(y_pred, axis=1)

# Calculate the accuracy by comparing the predicted labels to the true labels
accuracy = np.mean(y_pred_labels == y_test)
print(f"Test accuracy: {accuracy}")

