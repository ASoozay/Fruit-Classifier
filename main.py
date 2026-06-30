import tensorflow as tf
import numpy as np
import pathlib as pl

from PIL import Image
import os

root = "D:/Fruits Classification"

train_ds = tf.keras.utils.image_dataset_from_directory(
    "D:/Fruits Classification/train",
    image_size=(224,224),
    batch_size=64,
    color_mode="rgb"
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    "D:/Fruits Classification/valid",
    image_size=(224,224),
    batch_size=64,
    color_mode="rgb"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "D:/Fruits Classification/test",
    image_size = (224, 224),
    batch_size = 64,
    color_mode="rgb"
)

class_names = train_ds.class_names
print(train_ds.class_names)

normalization = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization(x), y))
valid_ds = valid_ds.map(lambda x, y: (normalization(x), y))
test_ds = test_ds.map(lambda x, y: (normalization(x), y))

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))

print("Making Model...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),

    tf.keras.layers.Conv2D(8,3,activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(16,3,activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32,3,activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(5,activation='softmax')
])
print("Model complete")
model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=20
)

model.save("fruit-modal2.keras")

def predictFruit(fruit, fruitPath, model):
    correct_fruit = 0
    imagePath = pl.Path(fruitPath)
    pathLength = 0
    for image in imagePath.iterdir():
        pathLength += 1
        img = tf.keras.utils.load_img(image, target_size=(224,224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = img_array / 255
        img_array = tf.expand_dims(img_array, 0)
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        if(predicted_class == fruit):
            correct_fruit += 1

    accuracy = (correct_fruit / pathLength)  * 100
    print("Accuracy for ", fruit, " : ", accuracy, "%")


predictFruit("Apple", "D:/Fruits Classification/test/Apple", model)
predictFruit("Banana", "D:/Fruits Classification/test/Banana", model)
predictFruit("Grape", "D:/Fruits Classification/test/Grape", model)
predictFruit("Mango", "D:/Fruits Classification/test/Mango", model)
predictFruit("Strawberry", "D:/Fruits Classification/test/Strawberry", model)


