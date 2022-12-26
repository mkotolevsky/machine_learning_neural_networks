from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


train_dir = 'C:/Users/user/Desktop/project_2/train_1'
# Каталог с данными для обучения
val_dir = 'C:/Users/user/Desktop/project_2/val_1'
# Каталог с данными для проверки
test_dir = 'C:/Users/user/Desktop/project_2/test_1'
# Каталог с данными для тестирования

img_width, img_height = 150, 150  # Размеры изображений
# Размерность тензора на основе изображения для входных данных в нейронную сеть backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)

epochs = 3  # Количество эпох
batch_size = 25  # Размер мини-выборки

nb_train_samples = 1540  # Количество изображений для обучения
nb_validation_samples = 332  # Количество изображений для проверки
nb_test_samples = 332  # Количество изображений для тестирования

num_classes = 4  # Количество классов изображений (+, -, *, /)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='sparse'
)

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='sparse'
)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='sparse'
)

history = model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size
)

val_scores = model.evaluate(val_generator)
test_scores = model.evaluate(test_generator)
print(f"Accuracy on validation data: {(val_scores[1]*100):2}")
print(f"Accuracy on test data: {(test_scores[1]*100):2}")
print("Number of images: ", nb_train_samples + nb_validation_samples + nb_test_samples)
list_epochs = range(1, epochs + 1)

plt.title('Model accuracy')
plt.plot(list_epochs, history.history['accuracy'], 'r', label='Training acc')
plt.plot(list_epochs, history.history['val_accuracy'], 'y', label='Validation acc')
plt.xlabel('epoch number')
plt.ylabel('accuracy')
plt.grid(visible='on')
plt.legend()
plt.figure()

plt.title('Model loss')
plt.plot(list_epochs, history.history['loss'], 'r', label='Training loss')
plt.plot(list_epochs, history.history['val_loss'], 'y', label='Validation loss')
plt.xlabel('epoch number')
plt.ylabel('loss')
plt.grid(visible='on')
plt.legend()
plt.show()
