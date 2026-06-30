import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print("Running prediction script...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    "D:/Fruits Classification/test",
    image_size = (224, 224),
    batch_size = 64,
    color_mode="rgb"
)
class_names = test_ds.class_names
print(class_names)

model = tf.keras.models.load_model("fruit-modal2.keras")

normalization = tf.keras.layers.Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (normalization(x), y))

def predictFruit(test_ds, model):
    y_true = []
    y_pred= []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)

        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    for i, fruit in enumerate(class_names):
        mask = (y_true == i)

        total = np.sum(mask)
        correct = np.sum(y_pred[mask] == i)

        print(f"{fruit}")
        print(f"  Total actual: {total}")
        print(f"  Correct predictions: {correct}")
        print(f"  Accuracy: {correct / total * 100:.1f}%")
        print()



def fruitPredictMatrix(test_ds, model):
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)

        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=class_names)
    disp.plot(cmap="Blues")
    plt.show()


predictFruit(test_ds, model)

fruitPredictMatrix(test_ds, model)

