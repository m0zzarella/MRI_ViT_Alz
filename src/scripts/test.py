import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

TEST_PATH = 'src/data/test'

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

model = tf.keras.models.load_model('vit_model.h5')

loss, accuracy = model.evaluate(test_generator)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
