import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

data = keras.datasets.mnist
print(data)

(X_train, y_train),(X_test, y_test) = data.load_data()

print(X_train.shape)
#(60000, 28, 28) 60k-no of datasets    28*28- grid size

plt.matshow(X_train[0])

# Flattening the X_train & X_test
flat_X_train = X_train.reshape(len(X_train),28*28)
flat_X_test = X_test.reshape(len(X_test),28*28)

print(flat_X_train.shape)
#(60000, 784)


model = keras.Sequential([
                          keras.layers.Dense(units=10,
                                             input_shape=(784,),
                                             activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(flat_X_train,y_train, epochs=5)

tf.config.run_functions_eagerly(True)
model.evaluate(flat_X_test,y_test)
y_pred = model.predict(flat_X_test)

print(y_pred[0])

print(np.argmax(y_pred[0]))
plt.matshow(X_test[0])
plt.show()
y_pred_label = [np.argmax(i) for i in y_pred]
print(y_pred_label[0])
print(accuracy_score(y_pred_label,y_test))
# Confusion Matrix
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred_label)
print(cm)

# Import the necessary libraries
import numpy as np

# Get the predicted labels for all test samples
y_pred_labels = np.argmax(y_pred, axis=1)

# Calculate the accuracy by comparing the predicted labels to the true labels
accuracy = np.mean(y_pred_labels == y_test)
print(f"Test accuracy: {accuracy}")

# Multiple Hidden Layes
'''model = keras.Sequential([
                          keras.layers.Dense(units=10,
                                  
                                                        input_shape=(784,),
                                             activation='sigmoid'),
                          # Second Hidden Layer
                          keras.layers.Dense(units=100,
                                             activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(flat_X_train,y_train, epochs=5)'''


'''model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),
                          keras.layers.Dense(units=100,
                                             activation='sigmoid'),
                          # Second Hidden Layer
                          keras.layers.Dense(units=100,
                                             activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train,y_train, epochs=5)'''


'''# Export the model

Pickling/Unpickling

Serialisation/De-serialisation

Dumping/Undumping'''