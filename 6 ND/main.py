import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix
import numpy as np

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

train_dir = "./mokymo_aibe"
test_dir = "./testavimo_aibe"

# Normalizavimas
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Nuskaitymas
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# print(train_data.class_indices)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# print(test_data.class_indices)

# Modelio parametrai
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(train_data.num_classes, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Modelio mokymas
history = model.fit(
    train_data,
    epochs=10,
    validation_data=test_data
)

# Modelio testavimas
test_loss, test_acc = model.evaluate(test_data)

# Modelio analizė
y_true = test_data.classes
y_pred = model.predict(test_data)
y_pred_clases = np.argmax(y_pred, axis=1)

confusion_m = confusion_matrix(y_true, y_pred_clases)

print("Modelis 1 - bazinis CNN")
print("Parametrai: IMG_SIZE=128x128, BATCH_SIZE=32, epochs=10, optimizer=adam")
print("Testavimo praradimas:", test_loss)
print("Tikslumas:", test_acc)
print(confusion_m)