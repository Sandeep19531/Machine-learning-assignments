#it is a supervised learning model
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist#dataset contains fashion images in grayscale format

(train_images, train_labels), (test_images, test_labels) = data.load_data() #dividing the complete data into two sets

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0 #images are in grayscale so the pixle will be between 0  and 255
test_images = test_images/255.0
"""training model :
            flatten: converts the matrix into in an array like structure to reduce complexity"""
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",metrics=["accuracy"])
#this model using loss to check the progress where system will learn from the mistakes it make
model.fit(train_images,train_labels, epochs=5) #epochs tell how many times the model is to be trained

test_loss, test_acc =model.evaluate(test_images, test_labels)

print("Tested accuracy =", test_acc)

"""prediction = model.predict(test_images)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual:" + class_names[test_labels[i]])
    plt.title("Prediction" + class_names[np.argmax(prediction[i])])
    plt.show()"""