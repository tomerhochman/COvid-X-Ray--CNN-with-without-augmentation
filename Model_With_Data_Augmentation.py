import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

import numpy as np


train_datagen = ImageDataGenerator(rotation_range=5,  # rotation
                                   width_shift_range=0.2,  # horizontal shift
                                   zoom_range=0.2,  # zoom
                                   horizontal_flip=True,  # horizontal flip
                                   brightness_range=[0.2, 0.8])  # brightness)


valid_datagen = ImageDataGenerator(rotation_range=5,  # rotation
                                   width_shift_range=0.2,  # horizontal shift
                                   zoom_range=0.2,  # zoom
                                   horizontal_flip=True,  # horizontal flip
                                   brightness_range=[0.2, 0.8])  # brightness)


test_datagen = ImageDataGenerator(rotation_range=5,  # rotation
                                  width_shift_range=0.2,  # horizontal shift
                                  zoom_range=0.2,  # zoom
                                  horizontal_flip=True,  # horizontal flip
                                  brightness_range=[0.2, 0.8])  # brightness)


train_path = r'C:\Users\1\OneDrive\Desktop\Pycharm_projects\Project2.0\splitted_data\train/'
valid_path = r"C:\Users\1\OneDrive\Desktop\Pycharm_projects\Project2.0\splitted_data\val/"
test_path = r"C:\Users\1\OneDrive\Desktop\Pycharm_projects\Project2.0\splitted_data\test/"


train_generator = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(224, 224), # resize to this size
    color_mode="rgb",  # for coloured images
    batch_size=32,  # number of images to extract from folder for every batch
    class_mode="categorical",  # classes to predict
    shuffle=True,
    seed=42)  # to make the result reproducible


valid_generator = valid_datagen.flow_from_directory(
    directory=valid_path,
    target_size=(224, 224), # resize to this size
    color_mode="rgb",  # for coloured images
    batch_size=32,  # number of images to extract from folder for every batch
    class_mode="categorical",  # classes to predict
    shuffle=True,
    seed=42)  # to make the result reproducible


test_generator = test_datagen.flow_from_directory(
    directory=test_path,
    target_size=(224, 224), # resize to this size
    color_mode="rgb",  # for coloured images
    batch_size=1,  # number of images to extract from folder for every batch
    class_mode='categorical',  # classes to predict
    seed=2020)  # to make the result reproducible


fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 15))

for i in range(4):
    # convert to unsigned integers for plotting
    image = next(train_generator)[0].astype('uint8')

    # changing size from (1, 200, 200, 3) to (200, 200, 3) for plotting the image
    image = np.squeeze(image)

    # plot raw pixel data
    ax[i].imshow(image)
    ax[i].axis('off')


# Model architecture
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.24))
model.add(Dense(2, activation='softmax'))
model.summary()


cce = CategoricalCrossentropy(from_logits=False)
optim = Adam()
model.compile(loss=cce, optimizer=optim, metrics=['accuracy'])


STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size


history = model.fit(train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=20
                    )


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.grid()
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train Loss', 'Val Loss'], loc='upper right')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train Accuracy', 'Val Accuracy'], loc='upper left')
plt.show()


STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
test_generator.reset()
pred = model.predict(test_generator,
                     steps=STEP_SIZE_TEST,
                     verbose=1)


predicted_classes_indices = np.argmax(pred, axis=1)
labels = train_generator.class_indices
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_classes_indices]
print(predictions)


# Model Analyzing
cm = confusion_matrix(y_true=test_generator.classes, y_pred=predicted_classes_indices)
disp = ConfusionMatrixDisplay(cm, display_labels=['covid', 'normal'])
disp.plot()


print(classification_report(y_true=test_generator.classes, y_pred=predicted_classes_indices))









