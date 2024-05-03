import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
from sklearn.linear_model import LogisticRegression
from keras.datasets import cifar10
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

subset_size = len(x_train) // 10
random_indices = np.random.choice(len(x_train), subset_size, replace=False)

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

cnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

cnn_model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

_, accuracy = cnn_model.evaluate(x_test, y_test)
print("CNN Accuracy:", accuracy)

class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Helper function to display images and their predicted labels
def display_images_with_predictions(model, x, y_true, class_labels):
    predictions = model.predict(x)
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(x[i])
        predicted_label = np.argmax(predictions[i])
        true_label = np.argmax(y_true[i])
        plt.title(f"Predicted: {class_labels[predicted_label]}\nTrue: {class_labels[true_label]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig(f"predictions_{model}.png")

display_images_with_predictions(cnn_model, x_test, y_test, class_labels)

from sklearn.metrics import confusion_matrix
import seaborn as sns

def display_confusion_matrix(model, x, y_true, class_labels):
    predictions = model.predict(x)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_true, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, vmin=0, vmax=1000)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    plt.savefig(f"confusion_matrix_{model}.png")

display_confusion_matrix(cnn_model, x_test, y_test, class_labels)