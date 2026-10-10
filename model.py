"""
model.py  -  Eye Open/Closed CNN training script (v2)

Folder structure needed (next to this file):

dataset/
  train/  Open/  Closed/
  test/   Open/  Closed/

Run:  python model.py
Output: models/cnnCat2.h5  (and models/cnnCat2.keras as backup)
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "dataset")
MODEL_DIR = os.path.join(BASE, "models")
IMG = 32                      
BATCH = 32
EPOCHS = 40                   
CLASSES = ["Closed", "Open"]  


def load(split, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, split),
        labels="inferred",
        label_mode="categorical",
        class_names=CLASSES,          
        color_mode="grayscale",
        image_size=(IMG, IMG),
        batch_size=BATCH,
        shuffle=shuffle,
    )


train_raw = load("train", True)
test_ds = load("test", False).prefetch(tf.data.AUTOTUNE)

augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.06),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.25),
    layers.RandomBrightness(0.25, value_range=(0, 255)),
])
train_ds = (train_raw
            .map(lambda x, y: (augment(x, training=True), y),
                 num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))

model = models.Sequential([
    layers.Input(shape=(IMG, IMG, 1)),
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dropout(0.4),
    layers.Dense(128, activation="relu"),
    layers.Dense(2, activation="softmax"),
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

cbs = [
    callbacks.EarlyStopping(monitor="val_accuracy", patience=10,
                            restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
]
model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=cbs)

loss, acc = model.evaluate(test_ds, verbose=0)
print(f"\nTest accuracy: {acc * 100:.2f}%")

y_true = np.concatenate([np.argmax(y, axis=1) for _, y in test_ds])
y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)
for i, name in enumerate(CLASSES):
    m = y_true == i
    print(f"  {name}: {(y_pred[m] == i).mean() * 100:.1f}% correct ({m.sum()} images)")

os.makedirs(MODEL_DIR, exist_ok=True)
model.save(os.path.join(MODEL_DIR, "cnnCat2.keras"))
print("Saved models/cnnCat2.keras")
try:
    model.save(os.path.join(MODEL_DIR, "cnnCat2.h5"))
    print("Saved models/cnnCat2.h5")
except Exception as e:
    print("Could not save .h5 (not a problem, .keras is saved):", e)
