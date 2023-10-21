from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical

from matplotlib import pyplot as plt
from keras.optimizers import SGD

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()


train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
train_images = train_images.astype('float32') / 255.0
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
test_images = test_images.astype('float32') / 255.0
# Applying the function to training set labels and testing set labels 
train_labels = to_categorical(train_labels, dtype="uint8")
test_labels = to_categorical(test_labels, dtype="uint8")

#opt = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=30, batch_size=64)
test_loss, test_acc = model.evaluate(test_images, test_labels)

model.save('model_V1_Adam.keras')










