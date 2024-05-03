import numpy as np
from sklearn.linear_model import LogisticRegression
from keras.datasets import cifar10
from keras.utils import to_categorical
import matplotlib.pyplot as plt

def display_images_with_predictions(model, x, y_true, class_labels):
    predictions = model.predict_proba(x)
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        image = x[i].reshape((32, 32, 3))  # Reshape the image back to (32, 32, 3)
        plt.imshow(image)
        predicted_label = np.argmax(predictions[i])
        true_label = np.argmax(y_true[i])
        plt.title(f"Predicted: {class_labels[predicted_label]}\nTrue: {class_labels[true_label]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig(f"predictions_{model}.png")

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

y_train = y_train.reshape(y_train.shape[0], -1)
y_test = y_test.reshape(y_test.shape[0], -1)

print(x_train.shape)
print(x_test.shape)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

logistic_model = LogisticRegression(verbose=True)
logistic_model.fit(x_train, y_train)

# Evaluate the model
accuracy = logistic_model.score(x_test, y_test)
print("Logistic Regression Accuracy:", accuracy)

from sklearn.svm import SVC
from sklearn.svm import LinearSVC

# SVM model
svm_model_rbf = SVC(kernel='rbf',probability=True, verbose=True)
svm_model_rbf.fit(x_train, y_train)

accuracy = svm_model_rbf.score(x_test, y_test)
print("SVM Accuracy:", accuracy)

import matplotlib.pyplot as plt

# Class labels
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# logistic_predictions = logistic_model.predict(x_test.reshape(x_test.shape[0], -1).astype('float32') / 255)
display_images_with_predictions(logistic_model, x_test, y_test, class_labels)

# svm_predictions = svm_model.predict(x_test.reshape(x_test.shape[0], -1).astype('float32') / 255)
display_images_with_predictions(svm_model_rbf, x_test, y_test, class_labels)


from sklearn.metrics import confusion_matrix
import seaborn as sns

def display_confusion_matrix(model, x, y_true, class_labels):
    predictions = model.predict(x)
    cm = confusion_matrix(y_true, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, vmin=0, vmax=1000)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    plt.savefig(f"confusion_matrix_{model}.png")

display_confusion_matrix(logistic_model, x_test, y_test.argmax(axis=1), class_labels)
display_confusion_matrix(svm_model_rbf, x_test, y_test.argmax(axis=1), class_labels)
