import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10

(x_train, y_train), (_, _) = cifar10.load_data()

# Class labels
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Number of images per class
class_counts = np.bincount(y_train.flatten())

plt.figure(figsize=(10, 5))
plt.bar(class_labels, class_counts)
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Distribution of CIFAR-10 Classes')
plt.xticks(rotation=45)
plt.show()

plt.savefig("distribution.png")

plt.figure(figsize=(10, 10))
for i in range(len(class_labels)):
    images = x_train[np.where(y_train == i)[0][:5]]
    for j in range(5):
        plt.subplot(10, 5, i * 5 + j + 1)
        plt.imshow(images[j])
        plt.title(class_labels[i])
        plt.axis('off')
plt.tight_layout()
plt.show()

plt.savefig("samples.png")

X_train = x_train.astype('float32') / 255.0

plt.figure(figsize=(10, 5))
plt.hist(X_train.flatten(), bins=50, color='blue', alpha=0.7)
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Pixel Value Distribution')
plt.show()

plt.savefig("pixels.png")
